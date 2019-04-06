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
from my_library.metrics import AttF1Measure, SentAcc


@Model.register("hotpot_decoupled")
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

        self._span_gate = SpanGate(encoding_dim, gate_threshold=0.3)
        self.qc_att = BiAttention(encoding_dim, dropout)
        self.qc_att_sp = BiAttention(encoding_dim, dropout)
        self.linear_1 = nn.Sequential(
                nn.Linear(encoding_dim, encoding_dim),
                nn.ReLU()
            )
        self.linear_2 = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim)
            #nn.ReLU()
        )
        self.self_att = BiAttention(encoding_dim, dropout, strong_sup=strong_sup)

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim * 3, 3)

        self._squad_metrics = SquadEmAndF1()

        self._f1_metrics = F1Measure(1)

        self._coref_f1_metric = AttF1Measure(0.1)

        self._sent_metrics = SentAcc()

        self._loss_trackers = {'loss': Average(),
                               'strong_sup_loss': Average()}

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                coref_mask: torch.FloatTensor = None
                ) -> Dict[str, torch.Tensor]:

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        decoupled_passage, spans_mask = convert_sequence_to_spans(embedded_passage, sentence_spans)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        encodeded_decoupled_passage = \
            self._phrase_layer_sp(
                decoupled_passage, spans_mask.view(batch_size * num_spans, -1))
        max_pooled_passage_embedding = torch.max(encodeded_decoupled_passage, 1)[0]
        # context_output_sp = convert_span_to_sequence(embedded_passage, encodeded_decoupled_passage, spans_mask)

        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()

        ques_output_sp = self._phrase_layer_sp(embedded_question, ques_mask)
        max_pooled_ques_embedding = torch.max(ques_output_sp, 1)[0]

        # modeled_passage_sp = self.qc_att_sp(context_output_sp, ques_output_sp, ques_mask)
        # modeled_passage_sp = self.linear_2(modeled_passage_sp)
        # modeled_passage_sp = self._modeling_layer_sp(modeled_passage_sp, context_mask)
        # Shape(spans_rep): (batch_size * num_spans, max_batch_span_width, embedding_dim)
        # Shape(spans_mask): (batch_size, num_spans, max_batch_span_width)
        # spans_rep_sp, spans_mask = convert_sequence_to_spans(modeled_passage_sp, sentence_spans)
        # Shape(gate_logit): (batch_size * num_spans, 2)
        # Shape(gate): (batch_size * num_spans, 1)
        # Shape(pred_sent_probs): (batch_size * num_spans, 2)
        # gate_logit = self._span_gate(spans_rep_sp, spans_mask)
        # print(max_pooled_passage_embedding.shape)
        # print(max_pooled_ques_embedding.view(batch_size, 1, -1).shape)
        gate_logit = torch.sum(self.linear_1(max_pooled_passage_embedding).view(batch_size, num_spans, -1) \
                     * self.linear_2(max_pooled_ques_embedding).view(batch_size, 1, -1), -1)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        sent_mask = (sent_labels >= 0).long()
        sent_labels = sent_labels * sent_mask
        # print(sent_labels)
        # print(gate_logit.shape)
        # print(gate_logit)
        strong_sup_loss = torch.mean(-torch.log(
            torch.sum(F.softmax(gate_logit) * sent_labels.float().view(batch_size, num_spans), dim=-1) + 1e-10))

        # strong_sup_loss = F.nll_loss(F.log_softmax(gate_logit, dim=-1).view(batch_size * num_spans, -1),
        #                              sent_labels.long().view(batch_size * num_spans), ignore_index=-1)

        gate = torch.argmax(gate_logit.view(batch_size, num_spans), -1)
        # gate = (gate >= 0.5).long().view(batch_size, num_spans)

        output_dict = {
            "gate": gate
        }

        loss = strong_sup_loss

        output_dict["loss"] = loss

        if metadata is not None:
            question_tokens = []
            passage_tokens = []
            sent_labels_list = []
            ids = []

            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                sent_labels_list.append(metadata[i]['sent_labels'])

            self._sent_metrics(gate, sent_labels)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['sent_labels'] = sent_labels_list
            output_dict['_id'] = ids

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        sent_acc = self._sent_metrics.get_metric(reset)
        metrics = {
                'evd_p': p,
                'evd_r': r,
                'evd_f1': evidence_f1_socre,
                'sent_acc': sent_acc
                }
        # for name, tracker in self._loss_trackers.items():
        #     metrics[name] = tracker.get_metric(reset).item()
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
    def __init__(self, span_dim, gate_threshold):
        super().__init__()
        self.modeled_gate = nn.Sequential(
            nn.Linear(span_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.gate_threshold = gate_threshold

    def forward(self,
                spans_tensor: torch.FloatTensor,
                spans_mask: torch.FloatTensor):

        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        # Shape: (batch_size * num_spans, embedding_dim)
        max_pooled_span_emb = torch.max(spans_tensor, dim=1)[0]
        # Shape: (batch_size * num_spans, 1)
        gate_logit = self.modeled_gate(max_pooled_span_emb)
        gate_logit = gate_logit.view(batch_size, num_spans)
        # gate_prob = F.softmax(gate_logit, dim=-1)
        # gate_prob = torch.chunk(gate_prob, 2, dim=-1)[1].view(batch_size * num_spans, -1)
        #
        # # print(torch.sum(positive_labels.view(-1) * gate.view(-1)))
        # dumb_gate = torch.full_like(gate_prob.view(batch_size * num_spans, -1).float(), self.gate_threshold)
        # new_gate = torch.cat([dumb_gate, gate_prob], dim=-1)

        # gate = (gate >= 0.3).long()
        # attended_span_embeddings = attended_span_embeddings * gate.unsqueeze(-1).float()
        return gate_logit
        # return gate_logit, gate_prob, new_gate


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