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
from allennlp.training.metrics import F1Measure, SquadEmAndF1, Average, CategoricalAccuracy
from my_library.metrics import AttF1Measure


@Model.register("wikihop_bert")
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

        self._categorical_acc = CategoricalAccuracy()

        self._coref_f1_metric = AttF1Measure(0.1)

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim, 3)

        self._loss_trackers = {'loss': Average()}

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor] = None,
                option: Dict[str, torch.LongTensor] = None,
                answer: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, option_num, option_length, _ = option['token_characters'].size()
        for k in option.keys():
            if k != 'token_characters':
                option[k] = option[k].view(batch_size*option_num, -1)
            else:
                option[k] = option[k].view(batch_size*option_num, option_length, -1)
        # option['token_characters'] = option['token_characters'].view(batch_size*option_num, option_length, -1)

        question_bert = self._text_field_embedder(question_passage)
        option_bert = self._text_field_embedder(option)
        question_cls_rep = torch.index_select(question_bert, 1, torch.tensor([0]).cuda())
        option_cls_rep = torch.index_select(option_bert, 1, torch.tensor([0]).cuda())
        option_cls_rep = option_cls_rep.view(batch_size, option_num, -1)

        opt_logits = torch.sum(option_cls_rep * question_cls_rep, dim=-1)

        opt_predicts = torch.argmax(opt_logits, 1)

        predict_loss = nll_loss(util.masked_log_softmax(opt_logits, None), answer)
        loss = predict_loss
        self._loss_trackers['loss'](loss)
        output_dict = {"loss": loss}

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['answer_texts'] = []
            question_tokens = []
            passage_tokens = []
            ids = []

            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_text'])
                passage_tokens.append(metadata[i]['original_passage'])
                # ids.append(metadata[i]['_id'])
                # passage_str = metadata[i]['original_passage']
                # offsets = metadata[i]['token_offsets']
                answer_text = metadata[i].get("answer_text")
                ans = answer[i]
                predict_ans = opt_predicts[i]
                # print(predict_ans, ans)
                output_dict['answer_texts'].append(answer_text)

                self._categorical_acc(opt_logits[i], ans)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['_id'] = ids

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        acc = self._categorical_acc.get_metric(reset)
        metrics = {
                'accuracy': acc,
                }
        # for name, tracker in self._loss_trackers.items():
        #     metrics[name] = tracker.get_metric(reset)
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
