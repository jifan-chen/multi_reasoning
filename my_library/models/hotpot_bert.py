from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, RegularizerApplicator
from allennlp.training.metrics import F1Measure, SquadEmAndF1, Average
from my_library.metrics import AttF1Measure


@Model.register("hotpot_bert_original")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        encoding_dim = text_field_embedder.get_output_dim()

        self._dropout = torch.nn.Dropout(p=dropout)

        self._squad_metrics = SquadEmAndF1()

        self._f1_metrics = F1Measure(1)

        self._coref_f1_metric = AttF1Measure(0.1)

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim, 3)

        self._loss_trackers = {'loss': Average(),
                               'start_loss': Average(),
                               'end_loss': Average(),
                               'type_loss': Average()}

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        question_passage_bert = self._text_field_embedder(question_passage)
        ques_passage_mask = util.get_text_field_mask(question_passage).float()

        batch_size = question_passage_bert.size()[0]

        span_start_logits = self.linear_start(question_passage_bert).squeeze(2) - 1e30 * (1 - ques_passage_mask)
        span_end_logits = self.linear_end(question_passage_bert).squeeze(2) - 1e30 * (1 - ques_passage_mask)
        cls_rep = torch.index_select(question_passage_bert, 1, torch.tensor([0]).cuda())
        output_type = torch.max(question_passage_bert, 1)[0]
        predict_type = self.linear_type(cls_rep.squeeze(dim=1))
        type_predicts = torch.argmax(predict_type, 1)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "best_span": best_span,
        }

        # Compute the loss for training.
        if span_start is not None:
            try:
                start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
                end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
                type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
                loss = start_loss + end_loss + type_loss
                self._loss_trackers['loss'](loss)
                self._loss_trackers['start_loss'](start_loss)
                self._loss_trackers['end_loss'](end_loss)
                self._loss_trackers['type_loss'](type_loss)
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
            ids = []
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
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['token_spans_sp'] = token_spans_sp
            output_dict['token_spans_sent'] = token_spans_sent
            output_dict['sent_labels'] = sent_labels_list
            output_dict['_id'] = ids

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        metrics = {
                'em': exact_match,
                'f1': f1_score,
                'evd_p': p,
                'evd_r': r,
                'evd_f1': evidence_f1_socre
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
            loss = torch.mean(-torch.log(torch.sum(weight_one * mask_sp.unsqueeze(1), dim=-1) + 1e-30))
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), loss, weight_one
        else:
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), weight_one
