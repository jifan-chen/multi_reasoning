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
from my_library.modules import PointerNetDecoder, AdpMemPointerNetDecoder, BiAttention


@Model.register("hotpot_legacy_rl")
class RLBidirectionalAttentionFlow(Model):
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
                 output_att_scores: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(RLBidirectionalAttentionFlow, self).__init__(vocab, regularizer)

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

        self._output_att_scores = output_att_scores

        encoding_dim = span_start_encoder.get_output_dim()

        self._span_gate = span_gate
        self.qc_att = BiAttention(encoding_dim, dropout)
        self.qc_att_sp = BiAttention(encoding_dim, dropout)
        self._gate_sent_encoder = gate_sent_encoder
        self._gate_self_attention_layer = gate_self_attention_layer

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim * 3, 3)

        self._squad_metrics = SquadEmAndF1_RT()

        self._f1_metrics = AttF1Measure(0.5, top_k=False)

        self._loss_trackers = {'loss': Average(),
                               'start_loss': Average(),
                               'end_loss': Average(),
                               'type_loss': Average(),
                               'rl_loss': Average()}

        self.evd_sup_acc_metric = ChainAccuracy()
        self.evd_ans_metric = Average()
        self.evd_beam_ans_metric = Average()

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                evd_chain_labels: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # In this model, we only take the first chain in ``evd_chain_labels`` for supervision
        evd_chain_labels = evd_chain_labels[:, 0]
        # there may be some instances that we can't find any evd chain for training
        # In that case, use the mask to ignore those instances
        evd_instance_mask = (evd_chain_labels[:, 0] != 0).float() if not evd_chain_labels is None else None

        # word + char embedding
        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        # mask
        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()

        # BiDAF for answer prediction
        ques_output = self._dropout(self._phrase_layer(embedded_question, ques_mask))
        context_output = self._dropout(self._phrase_layer(embedded_passage, context_mask))

        modeled_passage, _, qc_score = self.qc_att(context_output, ques_output, ques_mask)

        modeled_passage = self._modeling_layer(modeled_passage, context_mask)

        # BiDAF for chain prediction
        ques_output_sp = self._dropout(self._phrase_layer_sp(embedded_question, ques_mask))
        context_output_sp = self._dropout(self._phrase_layer_sp(embedded_passage, context_mask))

        modeled_passage_sp, _, qc_score_sp = self.qc_att_sp(context_output_sp, ques_output_sp, ques_mask)

        modeled_passage_sp = self._modeling_layer_sp(modeled_passage_sp, context_mask)

        # chain prediction
        # Shape(spans_rep): (batch_size * num_spans, max_batch_span_width, embedding_dim)
        # Shape(spans_mask): (batch_size, num_spans, max_batch_span_width)
        spans_rep_sp, spans_mask = convert_sequence_to_spans(modeled_passage_sp, sentence_spans)
        spans_rep, _ = convert_sequence_to_spans(modeled_passage, sentence_spans)
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

        # get the embeddings of the words of the sentences in the predicted chain
        spans_rep = spans_rep * gate.unsqueeze(-1)
        attended_sent_embeddings = convert_span_to_sequence(modeled_passage_sp, spans_rep, spans_mask)

        modeled_passage = attended_sent_embeddings + modeled_passage

        self_att_passage = self._self_attention_layer(modeled_passage, mask=context_mask)
        modeled_passage = modeled_passage + self_att_passage[0]
        self_att_score = self_att_passage[2]

        output_start = self._span_start_encoder(modeled_passage, context_mask)
        span_start_logits = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        output_end = torch.cat([modeled_passage, output_start], dim=2)
        output_end = self._span_end_encoder(output_end, context_mask)
        span_end_logits = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

        output_type = torch.cat([modeled_passage, output_end, output_start], dim=2)
        output_type = torch.max(output_type, 1)[0]
        # output_type = torch.max(self.rnn_type(output_type, context_mask), 1)[0]
        predict_type = self.linear_type(output_type)
        type_predicts = torch.argmax(predict_type, 1)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "best_span": best_span,
            "pred_sent_labels": gate.squeeze(1).view(batch_size, num_spans), #[B, num_span]
            "gate_probs": gate_probs.squeeze(1).view(batch_size, num_spans), #[B, num_span]
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
        evd_TP, evd_NP, evd_NT = self._f1_metrics(gate.squeeze(1).view(batch_size, num_spans),
                                                  sent_labels,
                                                  mask=gate_mask,
                                                  instance_mask=evd_instance_mask if self.training else None,
                                                  sum=False)
        print("TP:", evd_TP)
        print("NP:", evd_NP)
        print("NT:", evd_NT)
        evd_ps = np.array(evd_TP) / (np.array(evd_NP) + 1e-13)
        evd_rs = np.array(evd_TP) / (np.array(evd_NT) + 1e-13)
        evd_f1s = 2. * ((evd_ps * evd_rs) / (evd_ps + evd_rs + 1e-13))
        predict_mask = get_evd_prediction_mask(all_predictions.unsqueeze(1), eos_idx=0)[0]
        gold_mask = get_evd_prediction_mask(evd_chain_labels, eos_idx=0)[0]
        # ChainAccuracy defaults to take multiple predicted chains, so unsqueeze dim 1
        self.evd_sup_acc_metric(predictions=all_predictions.unsqueeze(1), gold_labels=evd_chain_labels,
                                predict_mask=predict_mask, gold_mask=gold_mask, instance_mask=evd_instance_mask)
        print("gold chain:", evd_chain_labels)
        predict_mask = predict_mask.float().squeeze(1)
        rl_loss = -torch.mean(torch.sum(all_logprobs * predict_mask * evd_instance_mask[:, None], dim=1))

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        # Compute before loss for rl
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['answer_texts'] = []
            question_tokens = []
            passage_tokens = []
            token_spans_sp = []
            token_spans_sent = []
            sent_labels_list = []
            evd_possible_chains = []
            ans_sent_idxs = []
            pred_chains_include_ans = []
            beam_pred_chains_include_ans = []
            ids = []
            ems = []
            f1s = []
            count_yes = 0
            count_no = 0
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                token_spans_sp.append(metadata[i]['token_spans_sp'])
                token_spans_sent.append(metadata[i]['token_spans_sent'])
                sent_labels_list.append(metadata[i]['sent_labels'])
                ids.append(metadata[i]['_id'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                if type_predicts[i] == 1:
                    best_span_string = 'yes'
                    count_yes += 1
                elif type_predicts[i] == 2:
                    best_span_string = 'no'
                    count_no += 1
                else:
                    predicted_span = tuple(best_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_span_string = passage_str[start_offset:end_offset]

                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                output_dict['answer_texts'].append(answer_texts)

                if answer_texts:
                    em, f1 = self._squad_metrics(best_span_string.lower(), answer_texts)
                    ems.append(em)
                    f1s.append(f1)

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
            output_dict['evd_possible_chains'] = evd_possible_chains
            output_dict['ans_sent_idxs'] = ans_sent_idxs
            output_dict['pred_chains_include_ans'] = pred_chains_include_ans
            output_dict['beam_pred_chains_include_ans'] = beam_pred_chains_include_ans
            output_dict['_id'] = ids

        # Compute the loss for training.
        if span_start is not None:
            try:
                start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
                end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
                type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
                loss = start_loss + end_loss + type_loss + rl_loss
                self._loss_trackers['loss'](loss)
                self._loss_trackers['start_loss'](start_loss)
                self._loss_trackers['end_loss'](end_loss)
                self._loss_trackers['type_loss'](type_loss)
                self._loss_trackers['rl_loss'](rl_loss)
                output_dict["loss"] = loss
            except RuntimeError:
                print('\n meta_data:', metadata)
                print(output_dict['_id'])
                print(span_start_logits.shape)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        p, r, evidence_f1_score = self._f1_metrics.get_metric(reset)
        ans_in_evd = self.evd_ans_metric.get_metric(reset)
        beam_ans_in_evd = self.evd_beam_ans_metric.get_metric(reset)
        metrics = {
                'em': exact_match,
                'f1': f1_score,
                'evd_p': p,
                'evd_r': r,
                'evd_f1': evidence_f1_score,
                'ans_in_evd': ans_in_evd,
                'beam_ans_in_evd': beam_ans_in_evd,
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


@Seq2SeqEncoder.register("span_gate")
class SpanGate(Seq2SeqEncoder):
    def __init__(self, span_dim,
                       max_decoding_steps=5, predict_eos=True, cell='lstm',
                       train_helper="sample", val_helper="beamsearch", beam_size=3,
                       aux_input_dim=None,#200,#None,
                       pass_label=False):
        super().__init__()
        self.evd_decoder = PointerNetDecoder(LinearMatrixAttention(span_dim, span_dim, "x,y,x*y"),
                                             memory_dim=span_dim,
                                             aux_input_dim=aux_input_dim,
                                             train_helper=train_helper,
                                             val_helper=val_helper,
                                             beam_size=beam_size,
                                             max_decoding_steps=max_decoding_steps,
                                             predict_eos=predict_eos,
                                             cell=cell)

    def forward(self,
                spans_tensor: torch.FloatTensor,
                spans_mask: torch.FloatTensor,
                question_tensor: torch.FloatTensor,
                question_mask: torch.FloatTensor,
                evd_chain_labels: torch.FloatTensor,
                self_att_layer: Seq2SeqEncoder,
                sent_encoder: Seq2SeqEncoder,
                get_all_beam: bool=False):

        print("spans_tensor", spans_tensor.shape)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        # Shape: (batch_size * num_spans, embedding_dim)
        max_pooled_span_emb = torch.max(spans_tensor, dim=1)[0]

        # self attention on spans representation
        # shape: (batch_size, num_spans, embedding_dim)
        max_pooled_span_emb = max_pooled_span_emb.view(batch_size, num_spans, spans_tensor.size(2))
        # shape: (batch_size, num_spans)
        max_pooled_span_mask = (torch.sum(spans_mask, dim=-1) >= 1).float()
        '''
        # shape: (batch_size, num_spans, embedding_dim)
        max_pooled_span_emb = sent_encoder(max_pooled_span_emb, max_pooled_span_mask)
        # shape: (batch_size, num_spans, embedding_dim)
        att_max_pooled_span_emb, _, att_score = self_att_layer(max_pooled_span_emb, max_pooled_span_mask)
        max_pooled_span_emb = max_pooled_span_emb + att_max_pooled_span_emb
        '''
        att_score = None

        # extract the final hidden states as the question vector
        # Shape: (batch_size, embedding_dim)
        question_emb = util.get_final_encoder_states(question_tensor, question_mask, True)

        # decode the most likely evidence path
        # shape (all_predictions): (batch_size, K, num_decoding_steps)
        # shape (all_logprobs): (batch_size, K, num_decoding_steps)
        # shape (seq_logprobs): (batch_size, K)
        # shape (final_hidden): (batch_size, K, decoder_output_dim)
        all_predictions, all_logprobs, seq_logprobs, final_hidden = self.evd_decoder(max_pooled_span_emb,
                                                                                     max_pooled_span_mask,
                                                                                     question_emb,
                                                                                     aux_input=None,#question_emb,#None
                                                                                     transition_mask=None,
                                                                                     labels=evd_chain_labels)
        print("all prediction:", all_predictions)

        # The selection order of each sentence. Set to -1 if not being chosen
        # shape: (batch_size, K, num_spans)
        _, beam, num_steps = all_predictions.size()
        orders = spans_tensor.new_ones((batch_size, beam, 1+num_spans)) * -1
        indices = util.get_range_vector(num_steps, util.get_device_of(spans_tensor)).\
                float().\
                unsqueeze(0).\
                unsqueeze(0).\
                expand(batch_size, beam, num_steps)
        orders.scatter_(2, all_predictions, indices)
        orders = orders[:, :, 1:]

        # For beamsearch, get the top one. For other helpers, just like squeeze
        if not get_all_beam:
            all_predictions = all_predictions[:, 0, :]
            all_logprobs = all_logprobs[:, 0, :]
            seq_logprobs = seq_logprobs[:, 0]
            final_hidden = final_hidden[:, 0, :]

        # build the gate. The dim is set to 1 + num_spans to account for the end embedding
        # shape: (batch_size, 1+num_spans) or (batch_size, K, 1+num_spans)
        if not get_all_beam:
            gate = spans_tensor.new_zeros((batch_size, 1+num_spans))
        else:
            gate = spans_tensor.new_zeros((batch_size, beam, 1+num_spans))
        gate.scatter_(-1, all_predictions, 1.)
        # remove the column for end embedding
        # shape: (batch_size, num_spans) or (batch_size, K, num_spans)
        gate = gate[..., 1:]
        #print("gate:", gate)
        #print("real num:", torch.sum(gate, dim=1))
        #print("seq probs:", torch.exp(seq_logprobs))

        # shape: (batch_size * num_spans, 1) or (batch_size * K * num_spans, 1)
        if not get_all_beam:
            gate = gate.reshape(batch_size * num_spans, 1)
        else:
            gate = gate.reshape(batch_size * beam * num_spans, 1)

        # The probability of each selected sentence being selected. If not selected, set to 0.
        # shape: (batch_size * num_spans, 1) or (batch_size * K * num_spans, 1)
        if not get_all_beam:
            gate_probs = spans_tensor.new_zeros((batch_size, 1+num_spans))
        else:
            gate_probs = spans_tensor.new_zeros((batch_size, beam, 1+num_spans))
        gate_probs.scatter_(-1, all_predictions, all_logprobs.exp())
        gate_probs = gate_probs[..., 1:]
        if not get_all_beam:
            gate_probs = gate_probs.reshape(batch_size * num_spans, 1)
        else:
            gate_probs = gate_probs.reshape(batch_size * beam * num_spans, 1)

        return all_predictions, all_logprobs, seq_logprobs, gate, gate_probs, max_pooled_span_mask, att_score, orders


@Seq2SeqEncoder.register("adpmem_span_gate")
class AdpMemSpanGate(Seq2SeqEncoder):
    def __init__(self, span_dim,
                       max_decoding_steps=5, predict_eos=True, cell='lstm',
                       train_helper="sample", val_helper="beamsearch", beam_size=3,
                       aux_input_dim=None,#200,#None,
                       pass_label=False):
        super().__init__()
        self.evd_decoder = AdpMemPointerNetDecoder(LinearMatrixAttention(span_dim, span_dim, "x,y,x*y"),
                                             LinearMatrixAttention(span_dim, span_dim, "x,y,x*y"),
                                             memory_dim=span_dim,
                                             aux_input_dim=aux_input_dim,
                                             train_helper=train_helper,
                                             val_helper=val_helper,
                                             beam_size=beam_size,
                                             max_decoding_steps=max_decoding_steps,
                                             predict_eos=predict_eos,
                                             cell=cell)
        self._pass_label = pass_label

    def forward(self,
                spans_tensor: torch.FloatTensor,
                spans_mask: torch.FloatTensor,
                question_tensor: torch.FloatTensor,
                question_mask: torch.FloatTensor,
                evd_chain_labels: torch.FloatTensor,
                self_att_layer: Seq2SeqEncoder,
                sent_encoder: Seq2SeqEncoder,
                get_all_beam: bool=False):

        print("spans_tensor", spans_tensor.shape)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        # shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        spans_tensor = spans_tensor.view(batch_size, num_spans, max_batch_span_width, spans_tensor.size(2))
        # shape: (batch_size, num_spans)
        max_pooled_span_mask = (torch.sum(spans_mask, dim=-1) >= 1).float()
        att_score = None

        # extract the final hidden states as the question vector
        # Shape: (batch_size, embedding_dim)
        question_emb = util.get_final_encoder_states(question_tensor, question_mask, True)

        # decode the most likely evidence path
        # shape (all_predictions): (batch_size, K, num_decoding_steps)
        # shape (all_logprobs): (batch_size, K, num_decoding_steps)
        # shape (seq_logprobs): (batch_size, K)
        # shape (final_hidden): (batch_size, K, decoder_output_dim)
        all_predictions, all_logprobs, seq_logprobs, final_hidden = self.evd_decoder(spans_tensor,
                                                                                     spans_mask,
                                                                                     question_emb,
                                                                                     aux_input=None,#question_emb,#None
                                                                                     transition_mask=None,
                                                                                     labels=evd_chain_labels)
        if self._pass_label:
            all_predictions = evd_chain_labels.long().unsqueeze(1)
            all_logprobs = torch.zeros_like(all_predictions).float()
        #print("batch:", batch_size)
        #print("predict num:", torch.sum((all_predictions > 0).float(), dim=1))
        print("all prediction:", all_predictions)

        # The selection order of each sentence. Set to -1 if not being chosen
        # shape: (batch_size, K, num_spans)
        _, beam, num_steps = all_predictions.size()
        orders = spans_tensor.new_ones((batch_size, beam, 1+num_spans)) * -1
        indices = util.get_range_vector(num_steps, util.get_device_of(spans_tensor)).\
                float().\
                unsqueeze(0).\
                unsqueeze(0).\
                expand(batch_size, beam, num_steps)
        orders.scatter_(2, all_predictions, indices)
        orders = orders[:, :, 1:]

        # For beamsearch, get the top one. For other helpers, just like squeeze
        if not get_all_beam:
            all_predictions = all_predictions[:, 0, :]
            all_logprobs = all_logprobs[:, 0, :]
            seq_logprobs = seq_logprobs[:, 0]
            final_hidden = final_hidden[:, 0, :]

        # build the gate. The dim is set to 1 + num_spans to account for the end embedding
        # shape: (batch_size, 1+num_spans) or (batch_size, K, 1+num_spans)
        if not get_all_beam:
            gate = spans_tensor.new_zeros((batch_size, 1+num_spans))
        else:
            gate = spans_tensor.new_zeros((batch_size, beam, 1+num_spans))
        gate.scatter_(-1, all_predictions, 1.)
        # remove the column for end embedding
        # shape: (batch_size, num_spans) or (batch_size, K, num_spans)
        gate = gate[..., 1:]
        #print("gate:", gate)
        #print("real num:", torch.sum(gate, dim=1))
        #print("seq probs:", torch.exp(seq_logprobs))

        # shape: (batch_size * num_spans, 1) or (batch_size * K * num_spans, 1)
        if not get_all_beam:
            gate = gate.reshape(batch_size * num_spans, 1)
        else:
            gate = gate.reshape(batch_size * beam * num_spans, 1)

        # The probability of each selected sentence being selected. If not selected, set to 0.
        # shape: (batch_size * num_spans, 1) or (batch_size * K * num_spans, 1)
        if not get_all_beam:
            gate_probs = spans_tensor.new_zeros((batch_size, 1+num_spans))
        else:
            gate_probs = spans_tensor.new_zeros((batch_size, beam, 1+num_spans))
        gate_probs.scatter_(-1, all_predictions, all_logprobs.exp())
        gate_probs = gate_probs[..., 1:]
        if not get_all_beam:
            gate_probs = gate_probs.reshape(batch_size * num_spans, 1)
        else:
            gate_probs = gate_probs.reshape(batch_size * beam * num_spans, 1)

        return all_predictions, all_logprobs, seq_logprobs, gate, gate_probs, max_pooled_span_mask, att_score, orders
