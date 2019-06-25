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
from allennlp.nn.util import get_range_vector, get_device_of
from my_library.models.utils import convert_sequence_to_spans, convert_span_to_sequence
from my_library.metrics import AttF1Measure, SquadEmAndF1_RT, PerStepInclusion, ChainAccuracy
from my_library.metrics.per_step_inclusion import Evd_Reward, get_evd_prediction_mask
from my_library.modules import PointerNetDecoder, BiAttention


@Model.register("hotpot_legacy_rl_finetune")
class FineTuneRLBidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 phrase_layer_sp: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 self_attention_layer: Seq2SeqEncoder,
                 gate_sent_encoder: Seq2SeqEncoder,
                 gate_self_attention_layer: Seq2SeqEncoder,
                 type_encoder: Seq2SeqEncoder,
                 modeling_layer: Seq2SeqEncoder,
                 modeling_layer_sp: Seq2SeqEncoder,
                 span_gate: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 strong_sup: bool = False,
                 strict_eos: bool = False,
                 account_trans: bool = False,
                 output_att_scores: bool = True,
                 weights_file: str = None,
                 ft_reward: str = "ans",
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(FineTuneRLBidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._phrase_layer = phrase_layer
        self._phrase_layer_sp = phrase_layer_sp

        self._dropout = torch.nn.Dropout(p=dropout)

        self._modeling_layer = modeling_layer
        self._modeling_layer_sp = modeling_layer_sp

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder
        self._type_encoder = type_encoder

        self._self_attention_layer = self_attention_layer

        self._strong_sup = strong_sup
        self._strict_eos = strict_eos
        self._account_trans = account_trans
        self._output_att_scores = output_att_scores
        self._ft_reward = ft_reward.split('+')

        encoding_dim = span_start_encoder.get_output_dim()

        self._span_gate = span_gate
        self.qc_att = BiAttention(encoding_dim, 0.0)
        self.qc_att_sp = BiAttention(encoding_dim, dropout)
        self._gate_sent_encoder = gate_sent_encoder
        self._gate_self_attention_layer = gate_self_attention_layer

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim * 3, 3)

        self._squad_metrics = SquadEmAndF1_RT()

        self._f1_metrics = AttF1Measure(0.5, top_k=False)

        self._reward_metric = PerStepInclusion(eos_idx=0)

        self._coref_f1_metric = AttF1Measure(0.1)

        self._loss_trackers = {'loss': Average(),
                               #'start_loss': Average(),
                               #'end_loss': Average(),
                               #'type_loss': Average(),
                               'rl_loss': Average()}
        if self._strong_sup:
            self._loss_trackers['coref_sup_loss'] = Average()

        if self._span_gate.evd_decoder._train_helper_type == 'teacher_forcing':
            raise ValueError("train_helper should not be teacher forcing during fine tune!")
        self.evd_sup_acc_metric = ChainAccuracy()
        self.evd_ans_metric = Average()
        self.evd_beam_ans_metric = Average()
        self.evd_beam2_ans_metric = Average()

        # load the weights
        if weights_file:
            model_state = torch.load(weights_file, map_location=util.device_mapping(0))
            self.load_state_dict(model_state)
            #self.cuda(0)
            for p in self._text_field_embedder.parameters():
                p.requires_grad = False

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

        #ques_output = self._dropout(self._phrase_layer(embedded_question, ques_mask))
        #context_output = self._dropout(self._phrase_layer(embedded_passage, context_mask))
        ques_output = self._phrase_layer(embedded_question, ques_mask)
        context_output = self._phrase_layer(embedded_passage, context_mask)

        modeled_passage, _, qc_score = self.qc_att(context_output, ques_output, ques_mask)
        modeled_passage = self._modeling_layer(modeled_passage, context_mask)

        ques_output_sp = self._dropout(self._phrase_layer_sp(embedded_question, ques_mask))
        context_output_sp = self._dropout(self._phrase_layer_sp(embedded_passage, context_mask))
        #ques_output_sp = self._phrase_layer_sp(embedded_question, ques_mask)
        #context_output_sp = self._phrase_layer_sp(embedded_passage, context_mask)

        modeled_passage_sp, _, qc_score_sp = self.qc_att_sp(context_output_sp, ques_output_sp, ques_mask)
        modeled_passage_sp = self._modeling_layer_sp(modeled_passage_sp, context_mask)
        # Shape(spans_rep): (batch_size * num_spans, max_batch_span_width, embedding_dim)
        # Shape(spans_mask): (batch_size, num_spans, max_batch_span_width)
        spans_rep_sp, spans_mask = convert_sequence_to_spans(modeled_passage_sp, sentence_spans)
        spans_rep, _ = convert_sequence_to_spans(modeled_passage, sentence_spans)
        # Shape(all_predictions): (batch_size, K, num_decoding_steps)
        # Shape(all_logprobs): (batch_size, K, num_decoding_steps)
        # Shape(seq_logprobs): (batch_size, K)
        # Shape(gate): (batch_size * K * num_spans, 1)
        # Shape(gate_probs): (batch_size * K * num_spans, 1)
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
                                 self._gate_sent_encoder,
                                 get_all_beam=True)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        beam_size = all_predictions.size(1)

        #last_sent_gate = get_last_sent_gate(all_predictions, num_spans, get_all_beam=True, eos_idx=0)

        # expand all the tensor to fit the beam size
        num_toks = modeled_passage.size(1)
        emb_dim = spans_rep.size(-1)
        spans_rep = spans_rep.reshape(batch_size, num_spans, max_batch_span_width, emb_dim)
        spans_rep = spans_rep.unsqueeze(1).expand(batch_size, beam_size, num_spans, max_batch_span_width, emb_dim)
        spans_rep = spans_rep.reshape(batch_size * beam_size * num_spans, max_batch_span_width, emb_dim)
        spans_mask = spans_mask[:, None, :, :].expand(batch_size, beam_size, num_spans, max_batch_span_width)
        spans_mask = spans_mask.reshape(batch_size * beam_size, num_spans, max_batch_span_width)
        #modeled_passage = modeled_passage.unsqueeze(1).expand(batch_size, beam_size, num_toks, emb_dim)
        #modeled_passage = modeled_passage.reshape(batch_size * beam_size, num_toks, emb_dim)
        #modeled_passage = modeled_passage.unsqueeze(1).expand(batch_size, beam_size, num_toks, emb_dim)
        context_mask = context_mask.unsqueeze(1).expand(batch_size, beam_size, num_toks)
        context_mask = context_mask.reshape(batch_size * beam_size, num_toks)
        se_mask = gate.expand(batch_size * beam_size * num_spans, max_batch_span_width).unsqueeze(-1)
        #se_mask = last_sent_gate.expand(batch_size * beam_size * num_spans, max_batch_span_width).unsqueeze(-1)
        se_mask = convert_span_to_sequence(modeled_passage_sp, se_mask, spans_mask).squeeze(-1)

        spans_rep = spans_rep * gate.unsqueeze(-1)
        attended_sent_embeddings = convert_span_to_sequence(modeled_passage_sp, spans_rep, spans_mask)

        modeled_passage = attended_sent_embeddings# + modeled_passage

        if self._strong_sup:
            self_att_passage = self._self_attention_layer(modeled_passage,
                                                          mask=context_mask,
                                                          mask_sp=coref_mask,
                                                          att_sup_metric=self._coref_f1_metric)
            modeled_passage = modeled_passage + self_att_passage[0]
            coref_sup_loss = self_att_passage[1]
            self_att_score = self_att_passage[2]
        else:
            self_att_passage = self._self_attention_layer(modeled_passage, mask=context_mask)
            modeled_passage = modeled_passage + self_att_passage[0]
            self_att_score = self_att_passage[2]

        output_start = self._span_start_encoder(modeled_passage, context_mask)
        span_start_logits = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask * se_mask)
        output_end = torch.cat([modeled_passage, output_start], dim=2)
        output_end = self._span_end_encoder(output_end, context_mask)
        span_end_logits = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask * se_mask)

        output_type = torch.cat([modeled_passage, output_end, output_start], dim=2)
        output_type = torch.max(output_type, 1)[0]
        # output_type = torch.max(self.rnn_type(output_type, context_mask), 1)[0]
        predict_type = self.linear_type(output_type)
        type_predicts = torch.argmax(predict_type, 1)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits.view(batch_size, beam_size, num_toks)[:, 0, :],
            "span_end_logits": span_end_logits.view(batch_size, beam_size, num_toks)[:, 0, :],
            "best_span": best_span.view(batch_size, beam_size, 2)[:, 0, :],
            "pred_sent_labels": gate.squeeze(1).view(batch_size, beam_size, num_spans)[:, 0, :], #[B, num_span]
            "gate_probs": gate_probs.squeeze(1).view(batch_size, beam_size, num_spans)[:, 0, :], #[B, num_span]
            "pred_sent_orders": orders, #[B, K, num_span]
        }
        if self._output_att_scores:
            if not qc_score is None:
                output_dict['qc_score'] = qc_score
            if not qc_score_sp is None:
                output_dict['qc_score_sp'] = qc_score_sp
            if not self_att_score is None:
                output_dict['self_attention_score'] = self_att_score
            if not g_att_score is None:
                output_dict['evd_self_attention_score'] = g_att_score


        # compute evd rl training metric, rewards, and loss
        print("sent label:")
        for b_label in np.array(sent_labels.cpu()):
            b_label = b_label == 1
            indices = np.arange(len(b_label))
            print(indices[b_label] + 1)
        evd_TP, evd_NP, evd_NT = self._f1_metrics(gate.squeeze(1).view(batch_size, beam_size, num_spans)[:, 0, :],
                                                  sent_labels,
                                                  mask=gate_mask,
                                                  instance_mask=evd_instance_mask if self.training else None,
                                                  sum=False)
        print("TP:", evd_TP)
        print("NP:", evd_NP)
        print("NT:", evd_NT)
        per_step_included, per_step_mask, eos_mask = self._reward_metric(all_predictions[:, :1, :], sent_labels,
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
        predict_mask = get_evd_prediction_mask(all_predictions[:, :1, :], eos_idx=0)[0]
        gold_mask = get_evd_prediction_mask(evd_chain_labels, eos_idx=0)[0]
        # default to take multiple predicted chains, so unsqueeze dim 1
        self.evd_sup_acc_metric(predictions=all_predictions[:, :1, :], gold_labels=evd_chain_labels,
                                predict_mask=predict_mask, gold_mask=gold_mask, instance_mask=evd_instance_mask)
        print("gold chain:", evd_chain_labels)

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        # Compute before loss for rl
        best_span = best_span.view(batch_size, beam_size, 2)
        if metadata is not None:
            output_dict['best_span_str'] = []
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
            beam2_pred_chains_include_ans = []
            ids = []
            ems = []
            f1s = []
            rb_ems = []
            ch_lens = []
            #count_yes = 0
            #count_no = 0
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
                beam_best_span_string = []
                beam_f1s = []
                beam_ems = []
                beam_rb_ems = []
                beam_ch_lens = []
                for b_idx in range(beam_size):
                    if type_predicts[i] == 1:
                        best_span_string = 'yes'
                        #count_yes += 1
                    elif type_predicts[i] == 2:
                        best_span_string = 'no'
                        #count_no += 1
                    else:
                        predicted_span = tuple(best_span[i, b_idx].detach().cpu().numpy())
                        start_offset = offsets[predicted_span[0]][0]
                        end_offset = offsets[predicted_span[1]][1]
                        best_span_string = passage_str[start_offset:end_offset]
                    beam_best_span_string.append(best_span_string)

                    if answer_texts:
                        em, f1 = self._squad_metrics(best_span_string.lower(), answer_texts)
                        beam_ems.append(em)
                        beam_f1s.append(f1)

                    rb_chain = [s_idx-1 for s_idx in metadata[i]['evd_possible_chains'][0] if s_idx > 0]
                    pd_chain = [s_idx-1 for s_idx in all_predictions[i, b_idx].detach().cpu().numpy() if s_idx > 0]
                    beam_rb_ems.append(float(rb_chain == pd_chain))
                    beam_ch_lens.append(float(len(pd_chain)))
                ems.append(beam_ems)
                f1s.append(beam_f1s)
                rb_ems.append(beam_rb_ems)
                ch_lens.append(beam_ch_lens)
                output_dict['best_span_str'].append(beam_best_span_string[0])

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
                    if any([any([pred_sent_orders[beam][s_idx-1] >= 0 for s_idx in metadata[i]['ans_sent_idxs']]) 
                                                                        for beam in range(2)]):
                        self.evd_beam2_ans_metric(1)
                        beam2_pred_chains_include_ans.append(1)
                    else:
                        self.evd_beam2_ans_metric(0)
                        beam2_pred_chains_include_ans.append(0)

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
        # RL Loss equals ``-log(P) * (R - baseline)``
        # Shape: (batch_size, num_decoding_steps)
        per_step_rs = Evd_Reward(per_step_included, per_step_mask, eos_mask, evd_rs, evd_f1s,
                                 strict_eos=self._strict_eos, account_trans=self._account_trans)
        per_step_rs = per_step_rs.to(all_logprobs.device)
        #print("per_step_rs:", per_step_rs)
        tot_rs = 0.
        if "ans" in self._ft_reward:
            ans_rs = seq_logprobs.new_tensor(f1s) # shape: (batch_size, beam_size)
            print('ans rs:', ans_rs)
            tot_rs = tot_rs + ans_rs
        if "rb" in self._ft_reward:
            rb_rs = seq_logprobs.new_tensor(rb_ems) # shape: (batch_size, beam_size)
            print('rb rs:', rb_rs)
            tot_rs = tot_rs + 0.7 * rb_rs
        if "len" in self._ft_reward:
            len_rs = seq_logprobs.new_tensor(ch_lens) # shape: (batch_size, beam_size)
            len_rs = (1. - len_rs / 5.) * (len_rs > 0).float()
            print("len rs:", len_rs)
            tot_rs = tot_rs + 0.7 * len_rs
        #rs_baseline = torch.mean(rs)
        rs_baseline = 0#torch.mean(tot_rs)
        tot_rs = tot_rs - rs_baseline
        rl_loss = -torch.mean(seq_logprobs * tot_rs)
        if span_start is not None:
            #try:
            #start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
            # self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            #end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
            # self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            # self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            #type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
            #loss = start_loss + end_loss + type_loss + rl_loss
            loss = rl_loss
            if self._strong_sup:
                #loss += coref_sup_loss
                self._loss_trackers['coref_sup_loss'](coref_sup_loss)
            #print('start_loss:{} end_loss:{} type_loss:{}'.format(start_loss,end_loss,type_loss))
            self._loss_trackers['loss'](loss)
            #self._loss_trackers['start_loss'](start_loss)
            #self._loss_trackers['end_loss'](end_loss)
            #self._loss_trackers['type_loss'](type_loss)
            self._loss_trackers['rl_loss'](rl_loss)
            output_dict["loss"] = loss

            #except RuntimeError:
            #    print('\n meta_data:', metadata)
            #    print(span_start_logits.shape)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        p_ = self._reward_metric.get_metric(reset)
        coref_p, coref_r, coref_f1_score = self._coref_f1_metric.get_metric(reset)
        ans_in_evd = self.evd_ans_metric.get_metric(reset)
        beam_ans_in_evd = self.evd_beam_ans_metric.get_metric(reset)
        beam2_ans_in_evd = self.evd_beam2_ans_metric.get_metric(reset)
        metrics = {
                'em': exact_match,
                'f1': f1_score,
                'evd_p': p,
                'evd_p_': p_,
                'evd_r': r,
                'evd_f1': evidence_f1_socre,
                'coref_p': coref_p,
                'coref_r': coref_r,
                'core_f1': coref_f1_score,
                'ans_in_evd': ans_in_evd,
                'beam_ans_in_evd': beam_ans_in_evd,
                'beam2_ans_in_evd': beam2_ans_in_evd,
                }
        for name, tracker in self._loss_trackers.items():
            metrics[name] = tracker.get_metric(reset).item()
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


def get_last_sent_gate(all_predictions, num_spans, get_all_beam, eos_idx=0):
    """ all_predictions: shape: (batch_size, K, num_decoding_steps)
    """
    batch_size = all_predictions.size(0)
    num_steps = all_predictions.size(-1)
    # shape: (batch_size, K)
    last_poses = torch.sum((all_predictions != eos_idx).float(), dim=-1) - 1
    # shape: (num_decoding_steps, )
    indices = get_range_vector(num_steps, get_device_of(all_predictions)).float()
    # shape: (batch_size, K, num_decoding_steps)
    mask = (indices.view(*([1]*(all_predictions.dim()-1)), num_steps) == last_poses.unsqueeze(-1)).float()
    # shape: (batch_size, K, num_decoding_steps)
    last_predictions = all_predictions.float() * mask
    print("last_predictions:", last_predictions)

    # build the last sent gate. The dim is set to 1 + num_spans to account for the end embedding
    # shape: (batch_size, 1+num_spans) or (batch_size, K, 1+num_spans)
    if not get_all_beam:
        gate = last_predictions.new_zeros((batch_size, 1+num_spans))
    else:
        beam = all_predictions.size(1)
        gate = last_predictions.new_zeros((batch_size, beam, 1+num_spans))
    gate.scatter_(-1, last_predictions.long(), 1.)
    # remove the column for end embedding
    # shape: (batch_size, num_spans) or (batch_size, K, num_spans)
    gate = gate[..., 1:]

    # shape: (batch_size * num_spans, 1) or (batch_size * K * num_spans, 1)
    if not get_all_beam:
        gate = gate.reshape(batch_size * num_spans, 1)
    else:
        gate = gate.reshape(batch_size * beam * num_spans, 1)
    return gate
