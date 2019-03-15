from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, SquadEmAndF1, Average
from my_library.models.utils import convert_sequence_to_spans, convert_span_to_sequence
from my_library.metrics import AttF1Measure


@Model.register("hotpot_legacy")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 phrase_layer_sp: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 self_attention_layer: Seq2SeqEncoder,
                 span_self_attentive_encoder: Seq2SeqEncoder,
                 type_encoder: Seq2SeqEncoder,
                 modeling_layer: Seq2SeqEncoder,
                 modeling_layer_sp: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 strong_sup: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

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
        self._span_self_attentive_encoder = span_self_attentive_encoder

        self._strong_sup = strong_sup

        encoding_dim = span_start_encoder.get_output_dim()

        self._span_gate = SpanGate(encoding_dim)
        self.qc_att = BiAttention(encoding_dim, dropout)
        self.qc_att_sp = BiAttention(encoding_dim, dropout)
        self.linear_1 = nn.Sequential(
                nn.Linear(encoding_dim * 4, encoding_dim),
                nn.ReLU()
            )
        self.linear_2 = nn.Sequential(
            nn.Linear(encoding_dim * 4, encoding_dim),
            nn.ReLU()
        )
        self.self_att = BiAttention(encoding_dim, dropout, strong_sup=strong_sup)

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim * 3, 3)

        self._squad_metrics = SquadEmAndF1()

        self._f1_metrics = F1Measure(1)

        self._coref_f1_metric = AttF1Measure(0.1)

        self._loss_trackers = {'loss': Average(),
                               'start_loss': Average(),
                               'end_loss': Average(),
                               'type_loss': Average(),
                               'strong_sup_loss': Average()}
        if self._strong_sup:
            self._loss_trackers['coref_sup_loss'] = Average()

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                # dep_mask: torch.IntTensor = None,
                coref_mask: torch.FloatTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()

        ques_output = self._dropout(self._phrase_layer(embedded_question, ques_mask))
        context_output = self._dropout(self._phrase_layer(embedded_passage, context_mask))

        modeled_passage = self.qc_att(context_output, ques_output, ques_mask)
        modeled_passage = self.linear_1(modeled_passage)
        modeled_passage = self._modeling_layer(modeled_passage, context_mask)

        ques_output_sp = self._dropout(self._phrase_layer_sp(embedded_question, ques_mask))
        context_output_sp = self._dropout(self._phrase_layer_sp(embedded_passage, context_mask))

        modeled_passage_sp = self.qc_att_sp(context_output_sp, ques_output_sp, ques_mask)
        modeled_passage_sp = self.linear_2(modeled_passage_sp)
        modeled_passage_sp = self._modeling_layer_sp(modeled_passage_sp, context_mask)
        # Shape(spans_rep): (batch_size * num_spans, max_batch_span_width, embedding_dim)
        # Shape(spans_mask): (batch_size, num_spans, max_batch_span_width)
        spans_rep_sp, spans_mask = convert_sequence_to_spans(modeled_passage_sp, sentence_spans)
        spans_rep, _ = convert_sequence_to_spans(modeled_passage, sentence_spans)
        # Shape(gate_logit): (batch_size * num_spans, 2)
        # Shape(gate): (batch_size * num_spans, 1)
        # Shape(pred_sent_probs): (batch_size * num_spans, 2)
        gate_logit, gate, pred_sent_probs = self._span_gate(spans_rep_sp, spans_mask)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()

        strong_sup_loss = F.nll_loss(F.log_softmax(gate_logit, dim=-1).view(batch_size * num_spans, -1),
                                     sent_labels.long().view(batch_size * num_spans), ignore_index=-1)

        # gate = (gate >= 0.1).long()
        spans_rep = spans_rep * gate.unsqueeze(-1).float()
        attended_sent_embeddings = convert_span_to_sequence(modeled_passage_sp, spans_rep, spans_mask)

        modeled_passage = attended_sent_embeddings + modeled_passage
        ''' No residual, Apply gate on coref_mask
        modeled_passage = attended_sent_embeddings
        gate_sent = gate.expand(batch_size * num_spans, max_batch_span_width)
        gate_sent = convert_span_to_sequence(modeled_passage_sp, gate_sent, spans_mask).squeeze(2)
        gated_context_mask = context_mask.long() * gate_sent
        '''

        if self._strong_sup:
            #self_att_passage = self._self_attention_layer(modeled_passage, context_mask)
            #self_att_passage = self._self_attention_layer(modeled_passage, context_mask, dep_mask)
            self_att_passage = self._self_attention_layer(modeled_passage, context_mask, coref_mask, self._coref_f1_metric)
            #self_att_passage = self._self_attention_layer(modeled_passage, gated_context_mask, coref_mask, self._coref_f1_metric)
            modeled_passage = modeled_passage + self_att_passage[0]
            coref_sup_loss = self_att_passage[1]
            self_att_score = self_att_passage[2]
        else:
            self_att_passage = self._self_attention_layer(modeled_passage, context_mask)
            modeled_passage = modeled_passage + self_att_passage[0]
            self_att_score = self_att_passage[2]
            # modeled_passage = modeled_passage + \
            #                   self.linear_2(self.self_att(modeled_passage, modeled_passage, context_mask))

        output_start = self._span_start_encoder(modeled_passage, context_mask)
        span_start_logits = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        output_end = torch.cat([modeled_passage, output_start], dim=2)
        output_end = self._span_end_encoder(output_end, context_mask)
        span_end_logits = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

        output_type = torch.cat([modeled_passage, output_end, output_start], dim=2)
        output_type = torch.max(output_type, 1)[0]
        predict_type = self.linear_type(output_type)
        type_predicts = torch.argmax(predict_type, 1)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "best_span": best_span,
            "self_attention_score": self_att_score,
            "gate": gate.view(1, -1)
        }

        # Compute the loss for training.
        if span_start is not None:
            try:
                start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
                end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
                type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
                # loss = start_loss + end_loss + type_loss
                loss = start_loss + end_loss + type_loss + strong_sup_loss
                # loss = strong_sup_loss
                if self._strong_sup:
                    loss += coref_sup_loss
                    self._loss_trackers['coref_sup_loss'](coref_sup_loss)
                self._loss_trackers['loss'](loss)
                self._loss_trackers['start_loss'](start_loss)
                self._loss_trackers['end_loss'](end_loss)
                self._loss_trackers['type_loss'](type_loss)
                self._loss_trackers['strong_sup_loss'](strong_sup_loss)
                output_dict["loss"] = loss

            except RuntimeError:
                print('\n meta_data:', metadata)
                print(span_start_logits.shape)

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['answer_texts'] = []
            question_tokens = []
            passage_tokens = []
            token_spans_sp = []
            token_spans_sent = []
            sent_labels_list = []
            coref_clusters = []
            ids = []
            count_yes = 0
            count_no = 0
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
                    self._squad_metrics(best_span_string, answer_texts)
            self._f1_metrics(pred_sent_probs, sent_labels.view(-1))
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['token_spans_sp'] = token_spans_sp
            output_dict['token_spans_sent'] = token_spans_sent
            output_dict['sent_labels'] = sent_labels_list
            output_dict['coref_clusters'] = coref_clusters
            output_dict['_id'] = ids

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        coref_p, coref_r, coref_f1_score = self._coref_f1_metric.get_metric(reset)
        metrics = {
                'em': exact_match,
                'f1': f1_score,
                'evd_p': p,
                'evd_r': r,
                'evd_f1': evidence_f1_socre,
                'coref_p': coref_p,
                'coref_r': coref_r,
                'core_f1': coref_f1_score,
                }
        for name, tracker in self._loss_trackers.items():
            metrics[name] = tracker.get_metric(reset).item()
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


class SpanGate(nn.Module):
    def __init__(self, span_dim):
        super().__init__()
        self.modeled_gate = nn.Sequential(
            nn.Linear(span_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self,
                spans_tensor: torch.FloatTensor,
                spans_mask: torch.FloatTensor):

        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        # Shape: (batch_size * num_spans, embedding_dim)
        max_pooled_span_emb = torch.max(spans_tensor, dim=1)[0]
        # Shape: (batch_size * num_spans, 2)
        gate_logit = self.modeled_gate(max_pooled_span_emb)
        gate_prob = F.softmax(gate_logit, dim=-1)
        gate_prob = torch.chunk(gate_prob, 2, dim=-1)[1].view(batch_size * num_spans, -1)

        # positive_labels = span_labels.long() * has_span_mask.long().squeeze()
        # negative_labels = (1 - span_labels.long()) * has_span_mask.long().squeeze()
        # positive_num = torch.sum(positive_labels)
        # negative_num = torch.sum(negative_labels)

        # print(positive_labels, negative_labels)

        # print(positive_num, negative_num)
        # print(torch.sum(gate.float().view(-1) * positive_labels.float().view(-1)) / positive_num.float())
        # print(torch.sum(gate.float().view(-1) * negative_labels.float().view(-1)) / negative_num.float())

        # print(torch.sum(positive_labels.view(-1) * gate.view(-1)))
        dumb_gate = torch.full_like(gate_prob.view(batch_size * num_spans, -1).float(), 0.3)
        new_gate = torch.cat([dumb_gate, gate_prob], dim=-1)

        # gate = (gate >= 0.3).long()
        # attended_span_embeddings = attended_span_embeddings * gate.unsqueeze(-1).float()

        return gate_logit, gate_prob, new_gate


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout, strong_sup=False):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))
        self.strong_sup = strong_sup

    def forward(self, input, memory, mask, mask_sp=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        if self.strong_sup:
            # print(att.shape, mask_sp.shape)
            loss = torch.mean(-torch.log(torch.sum(weight_one * mask_sp.unsqueeze(1), dim=-1) + 1e-30))
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), loss
        else:
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)