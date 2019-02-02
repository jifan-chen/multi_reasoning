from torch.autograd import Variable
import numpy as np
from torch.nn.utils import rnn

import logging
from typing import Any, Dict, List, Optional
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder, MatrixAttention
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.matrix_attention import bilinear_matrix_attention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1


@Model.register("hotpot_legacy")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 type_encoder: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.word_dim = 300
        self._phrase_layer = phrase_layer
        self._dropout = torch.nn.Dropout(p=dropout)
        self.rnn_start = span_start_encoder
        self.rnn_end = span_end_encoder
        self.rnn_type = type_encoder

        self.hidden = 80

        char_hidden = 100
        hidden = 80
        keep_prob = 0.8
        self.rnn = EncoderRNN(char_hidden + self.word_dim, hidden, 1, True, True, 1-keep_prob, False)

        self.qc_att = BiAttention(hidden*2, 1-keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(hidden*8, hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(hidden, hidden, 1, False, True, 1-keep_prob, False)
        self.self_att = BiAttention(hidden*2, 1-keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(hidden*8, hidden),
                nn.ReLU()
            )

        self.linear_start = nn.Linear(hidden*2, 1)

        self.linear_end = nn.Linear(hidden*2, 1)

        self.linear_type = nn.Linear(hidden*2, 3)

        self.cache_S = 0
        self._squad_metrics = SquadEmAndF1()

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # print('y1:', span_start.squeeze())
        # print('y2:', span_end.squeeze())
        # print('q_type:', q_type)

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        batch_size = embedded_question.size(0)
        context_lens = embedded_passage.size(1)
        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()
        # print('context_length:', torch.sum(context_mask,dim=-1))
        ques_output = self._dropout(self._phrase_layer(embedded_question, ques_mask))
        context_output = self._dropout(self._phrase_layer(embedded_passage, context_mask))

        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        # output_t = self.rnn_2(output, context_lens)
        # output_t = self.self_att(output_t, output_t, context_mask)
        # output_t = self.linear_2(output_t)
        #
        # output = output + output_t

        output_start = self.rnn_start(output, context_mask)
        span_start_logits = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        output_end = torch.cat([output, output_start], dim=2)
        output_end = self.rnn_end(output_end, context_mask)
        span_end_logits = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.rnn_type(output_type, context_mask), 1)[0]
        predict_type = self.linear_type(output_type)
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
                # print(span_start_logits.shape)
                # print(span_end_logits.shape)
                # print(context_mask)
                # print(q_type)
                start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
                # self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
                end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
                # self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
                # self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
                # print('q_type:', q_type)
                type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
                loss = start_loss + end_loss + type_loss
                print('start_loss:{} end_loss:{} type_loss:{}'.format(start_loss,end_loss,type_loss))
                output_dict["loss"] = loss

            except RuntimeError:
                print('\n meta_data:', metadata)
                print(span_start_logits.shape)

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            count_yes = 0
            count_no = 0
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
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
                # print('type:', type_predicts[i])
                # print('answer_text:', answer_texts, 'predict:', best_span_string)
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            # print('yes:', count_yes)
            # print('no:', count_no)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'em': exact_match,
                'f1': f1_score,
                }

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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))
