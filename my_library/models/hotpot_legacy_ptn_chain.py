from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, SquadEmAndF1, Average
from my_library.models.utils import convert_sequence_to_spans, convert_span_to_sequence
from my_library.metrics import AttF1Measure, SquadEmAndF1_RT, PerStepInclusion, ChainAccuracy
from my_library.metrics.per_step_inclusion import Evd_Reward, get_evd_prediction_mask
from my_library.modules import PointerNetDecoder, BiAttention


@Model.register("hotpot_legacy_ptn_chain")
class PTNChainBidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer_sp: Seq2SeqEncoder,
                 gate_sent_encoder: Seq2SeqEncoder,
                 gate_self_attention_layer: Seq2SeqEncoder,
                 modeling_layer_sp: Seq2SeqEncoder,
                 span_gate: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 strict_eos: bool = False,
                 account_trans: bool = False,
                 output_att_scores: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(PTNChainBidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._phrase_layer_sp = phrase_layer_sp

        self._dropout = torch.nn.Dropout(p=dropout)

        self._modeling_layer_sp = modeling_layer_sp

        self._strict_eos = strict_eos
        self._account_trans = account_trans
        self._output_att_scores = output_att_scores

        encoding_dim = phrase_layer_sp.get_output_dim()

        self._span_gate = span_gate
        self.qc_att_sp = BiAttention(encoding_dim, dropout)
        self._gate_sent_encoder = gate_sent_encoder
        self._gate_self_attention_layer = gate_self_attention_layer

        self._f1_metrics = AttF1Measure(0.5, top_k=False)

        self._reward_metric = PerStepInclusion(eos_idx=0)

        self._loss_trackers = {'loss': Average(),
                               'rl_loss': Average()}

        if self._span_gate.evd_decoder._train_helper_type == 'teacher_forcing':
            self._evd_train_type = 'supervised'
            self.evd_sup_acc_metric = ChainAccuracy()
            self.evd_ans_metric = Average()
            self.evd_beam_ans_metric = Average()
        else:
            self._evd_train_type = 'rl'

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                evd_chain_labels: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                # dep_mask: torch.IntTensor = None,
                coref_mask: torch.FloatTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # In this model, we only take the first chain in ``evd_chain_labels`` for supervision
        evd_chain_labels = evd_chain_labels[:, 0]
        # there may be some instances that we can't find any evd chain for training
        # In that case, use the mask to ignore those instances
        evd_instance_mask = (evd_chain_labels[:, 0] != 0).float() if not evd_chain_labels is None else None

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()

        #embedded_question = self._dropout(embedded_question)
        #embedded_passage = self._dropout(embedded_passage)

        ques_output_sp = self._dropout(self._phrase_layer_sp(embedded_question, ques_mask))
        context_output_sp = self._dropout(self._phrase_layer_sp(embedded_passage, context_mask))
        #ques_output_sp = self._phrase_layer_sp(embedded_question, ques_mask)
        #context_output_sp = self._phrase_layer_sp(embedded_passage, context_mask)

        modeled_passage_sp, _, qc_score_sp = self.qc_att_sp(context_output_sp, ques_output_sp, ques_mask)
        modeled_passage_sp = self._modeling_layer_sp(modeled_passage_sp, context_mask)
        # Shape(spans_rep): (batch_size * num_spans, max_batch_span_width, embedding_dim)
        # Shape(spans_mask): (batch_size, num_spans, max_batch_span_width)
        spans_rep_sp, spans_mask = convert_sequence_to_spans(modeled_passage_sp, sentence_spans)
        # Shape(all_predictions): (batch_size, num_decoding_steps)
        # Shape(all_logprobs): (batch_size, num_decoding_steps)
        # Shape(seq_logprobs): (batch_size,)
        # Shape(gate): (batch_size * num_spans, 1)
        # Shape(gate_probs): (batch_size * num_spans, 1)
        # Shape(gate_mask): (batch_size, num_spans)
        # Shape(g_att_score): (batch_size, num_heads, num_spans, num_spans)
        # Shape(orders): (batch_size, K, num_spans)
        all_predictions,    \
        all_logprobs,       \
        seq_logprobs,       \
        gate,               \
        gate_probs,         \
        gate_mask,          \
        g_att_score,        \
        orders = self._span_gate(spans_rep_sp, spans_mask,
                                 ques_output_sp, ques_mask,
                                 evd_chain_labels,
                                 self._gate_self_attention_layer,
                                 self._gate_sent_encoder)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()

        output_dict = {
            "pred_sent_labels": gate.squeeze(1).view(batch_size, num_spans), #[B, num_span]
            "gate_probs": gate_probs.squeeze(1).view(batch_size, num_spans), #[B, num_span]
            "pred_sent_orders": orders, #[B, K, num_span]
        }
        if self._output_att_scores:
            if not qc_score_sp is None:
                output_dict['qc_score_sp'] = qc_score_sp
            if not g_att_score is None:
                output_dict['evd_self_attention_score'] = g_att_score


        # compute evd rl training metric, rewards, and loss
        print("sent label:")
        for b_label in np.array(sent_labels.cpu()):
            b_label = b_label == 1
            indices = np.arange(len(b_label))
            print(indices[b_label] + 1)
        evd_TP, evd_NP, evd_NT = self._f1_metrics(gate.squeeze(1).view(batch_size, num_spans),
                                                  sent_labels,
                                                  mask=gate_mask,
                                                  instance_mask=evd_instance_mask if self.training else None,
                                                  sum=False)
        print("TP:", evd_TP)
        print("NP:", evd_NP)
        print("NT:", evd_NT)
        per_step_included, per_step_mask, eos_mask = self._reward_metric(all_predictions.unsqueeze(1), sent_labels,
                                                                         gate_mask,
                                                                         instance_mask=evd_instance_mask if self.training else None)
        per_step_included, per_step_mask, eos_mask = per_step_included.squeeze(1), per_step_mask.squeeze(1), eos_mask.squeeze(1)
        #print("per_step_included:", per_step_included)
        #print("per_step_mask:", per_step_mask)
        #print("eos_mask:", eos_mask)
        evd_ps = np.array(evd_TP) / (np.array(evd_NP) + 1e-13)
        evd_rs = np.array(evd_TP) / (np.array(evd_NT) + 1e-13)
        evd_f1s = 2. * ((evd_ps * evd_rs) / (evd_ps + evd_rs + 1e-13))
        #print("evd_f1s:", evd_f1s)
        if self._evd_train_type == 'supervised':
            predict_mask = get_evd_prediction_mask(all_predictions.unsqueeze(1), eos_idx=0)[0]
            gold_mask = get_evd_prediction_mask(evd_chain_labels, eos_idx=0)[0]
            # default to take multiple predicted chains, so unsqueeze dim 1
            self.evd_sup_acc_metric(predictions=all_predictions.unsqueeze(1), gold_labels=evd_chain_labels,
                                    predict_mask=predict_mask, gold_mask=gold_mask, instance_mask=evd_instance_mask)
            print("gold chain:", evd_chain_labels)
        if self._evd_train_type == 'rl':
            # RL Loss equals ``-log(P) * (R - baseline)``
            # Shape: (batch_size, num_decoding_steps)
            per_step_rs = Evd_Reward(per_step_included, per_step_mask, eos_mask, evd_rs, evd_f1s,
                                     strict_eos=self._strict_eos, account_trans=self._account_trans)
            per_step_rs = per_step_rs.to(all_logprobs.device)
            #print("per_step_rs:", per_step_rs)
            rl_loss = -torch.mean(torch.sum(all_logprobs * per_step_rs, dim=1))
        elif self._evd_train_type == 'supervised':
            per_step_mask = per_step_mask.to(all_logprobs.device)
            rl_loss = -torch.mean(torch.sum(all_logprobs * per_step_mask * evd_instance_mask[:, None], dim=1))

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        # Compute before loss for rl
        if metadata is not None:
            output_dict['answer_texts'] = []
            question_tokens = []
            passage_tokens = []
            token_spans_sp = []
            token_spans_sent = []
            sent_labels_list = []
            coref_clusters = []
            evd_possible_chains = []
            ans_sent_idxs = []
            pred_chains_include_ans = []
            beam_pred_chains_include_ans = []
            ids = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                token_spans_sp.append(metadata[i]['token_spans_sp'])
                token_spans_sent.append(metadata[i]['token_spans_sent'])
                sent_labels_list.append(metadata[i]['sent_labels'])
                coref_clusters.append(metadata[i]['coref_clusters'])
                ids.append(metadata[i]['_id'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                answer_texts = metadata[i].get('answer_texts', [])
                output_dict['answer_texts'].append(answer_texts)

                # shift sentence indice back
                evd_possible_chains.append([s_idx-1 for s_idx in metadata[i]['evd_possible_chains'][0] if s_idx > 0])
                ans_sent_idxs.append([s_idx-1 for s_idx in metadata[i]['ans_sent_idxs']])
                print("ans_sent_idxs:", metadata[i]['ans_sent_idxs'])
                if len(metadata[i]['ans_sent_idxs']) > 0:
                    pred_sent_orders = orders[i].detach().cpu().numpy()
                    if any([pred_sent_orders[0][s_idx-1] >= 0 for s_idx in metadata[i]['ans_sent_idxs']]):
                        self.evd_ans_metric(1)
                        pred_chains_include_ans.append(1)
                    else:
                        self.evd_ans_metric(0)
                        pred_chains_include_ans.append(0)
                    if any([any([pred_sent_orders[beam][s_idx-1] >= 0 for s_idx in metadata[i]['ans_sent_idxs']]) 
                                                                        for beam in range(len(pred_sent_orders))]):
                        self.evd_beam_ans_metric(1)
                        beam_pred_chains_include_ans.append(1)
                    else:
                        self.evd_beam_ans_metric(0)
                        beam_pred_chains_include_ans.append(0)

            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['token_spans_sp'] = token_spans_sp
            output_dict['token_spans_sent'] = token_spans_sent
            output_dict['sent_labels'] = sent_labels_list
            output_dict['coref_clusters'] = coref_clusters
            output_dict['evd_possible_chains'] = evd_possible_chains
            output_dict['ans_sent_idxs'] = ans_sent_idxs
            output_dict['pred_chains_include_ans'] = pred_chains_include_ans
            output_dict['beam_pred_chains_include_ans'] = beam_pred_chains_include_ans
            output_dict['_id'] = ids

        # Compute the loss for training.
        if span_start is not None:
            try:
                loss = rl_loss
                #print('start_loss:{} end_loss:{} type_loss:{}'.format(start_loss,end_loss,type_loss))
                self._loss_trackers['loss'](loss)
                self._loss_trackers['rl_loss'](rl_loss)
                output_dict["loss"] = loss

            except RuntimeError:
                print('\n meta_data:', metadata)
                print(output_dict['_id'])
                print(loss)
                print(span_start_logits.shape)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        p_ = self._reward_metric.get_metric(reset)
        ans_in_evd = self.evd_ans_metric.get_metric(reset)
        beam_ans_in_evd = self.evd_beam_ans_metric.get_metric(reset)
        metrics = {
                'evd_p': p,
                'evd_p_': p_,
                'evd_r': r,
                'evd_f1': evidence_f1_socre,
                'ans_in_evd': ans_in_evd,
                'beam_ans_in_evd': beam_ans_in_evd,
                }
        for name, tracker in self._loss_trackers.items():
            metrics[name] = tracker.get_metric(reset).item()
        if self._evd_train_type == 'supervised':
            evd_sup_acc = self.evd_sup_acc_metric.get_metric(reset)
            metrics['evd_sup_acc'] = evd_sup_acc
        return metrics

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span
