from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder, MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

from my_library.modules.self_attentive_sentence_encoder import SelfAttentiveSpanExtractor


@Model.register("hotpot_legacy")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 self_attention_layer: Seq2SeqEncoder,
                 span_self_attentive_encoder: Seq2SeqEncoder,
                 type_encoder: Seq2SeqEncoder,
                 modeling_layer: Seq2SeqEncoder,
                 matrix_attention: MatrixAttention,
                 dropout: float = 0.2,
                 strong_sup: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        self._dropout = torch.nn.Dropout(p=dropout)
        self._modeling_layer = modeling_layer
        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder
        self._type_encoder = type_encoder
        self._self_attention_layer = self_attention_layer
        self._span_self_attentive_encoder = span_self_attentive_encoder
        self._matrix_attention = matrix_attention
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(modeling_layer.get_output_dim(),
                                                                    span_self_attentive_encoder)
        self._strong_sup = strong_sup

        encoding_dim = span_start_encoder.get_output_dim()
        hidden = 100

        self.qc_att = BiAttention(encoding_dim, dropout)
        self.linear_1 = nn.Sequential(
                nn.Linear(encoding_dim * 4, encoding_dim),
                nn.ReLU()
            )

        self.self_att = BiAttention(encoding_dim, dropout, strong_sup=strong_sup)
        self.linear_2 = nn.Sequential(
                nn.Linear(encoding_dim * 4, encoding_dim),
                nn.ReLU()
            )

        self.modeled_gate = nn.Sequential(
            nn.Linear(encoding_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim * 3, 3)

        self._squad_metrics = SquadEmAndF1()

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                dep_mask: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        batch_size = embedded_question.size(0)
        context_lens = embedded_passage.size(1)
        ques_mask = util.get_text_field_mask(question).float()
        context_mask = util.get_text_field_mask(passage).float()
        # print('context_mask:', context_mask.shape)
        # print('context_length:', torch.sum(context_mask, dim=-1))
        ques_output = self._dropout(self._phrase_layer(embedded_question, ques_mask))
        context_output = self._dropout(self._phrase_layer(embedded_passage, context_mask))

        modeled_passage = self.qc_att(context_output, ques_output, ques_mask)
        modeled_passage = self.linear_1(modeled_passage)
        modeled_passage = self._modeling_layer(modeled_passage, context_mask)

        attended_sent_embeddings = self._attentive_span_extractor(modeled_passage, sentence_spans)
        modeled_passage = modeled_passage + attended_sent_embeddings

        # print(attended_sent_embeddings.shape)
        # print(gate[0])
        # print(sp_mask[0])

        # for p, q in zip(gate[0], sp_mask[0]):
        #     if q == 1.0:
        #         print(p, q)
        # print(torch.chunk(gate, 2, dim=-1)[0])

        if self._strong_sup:
            self_att_passage = self._self_attention_layer(modeled_passage, context_mask, sp_mask)
            modeled_passage = modeled_passage + self_att_passage[0]
            strong_sup_loss = self_att_passage[1]
        else:
            pass
            # modeled_passage = modeled_passage + \
            #                   self.linear_2(self.self_att(modeled_passage, modeled_passage, context_mask))

        # gate_logit = self.modeled_gate(modeled_passage)
        #
        # gate = F.softmax(gate_logit, dim=-1)
        # nll_loss = nn.NLLLoss()
        # strong_sup_loss1 = nll_loss(F.log_softmax(gate_logit, dim=-1).view(batch_size * context_lens, -1),
        #                             sp_mask.long().view(batch_size * context_lens))
        # print('\n strong_sup_loss1:', strong_sup_loss1)
        # modeled_passage = torch.chunk(gate, 2, dim=-1)[1] * modeled_passage + modeled_passage

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
        }

        # Compute the loss for training.
        if span_start is not None:
            try:
                start_loss = nll_loss(util.masked_log_softmax(span_start_logits, None), span_start.squeeze(-1))
                # self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
                end_loss = nll_loss(util.masked_log_softmax(span_end_logits, None), span_end.squeeze(-1))
                # self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
                # self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
                type_loss = nll_loss(util.masked_log_softmax(predict_type, None), q_type)
                loss = start_loss + end_loss + type_loss
                # loss = start_loss + end_loss + type_loss + (strong_sup_loss1 * 0.2)
                if self._strong_sup:
                    loss += strong_sup_loss
                    print('\n strong_sup_loss:', strong_sup_loss)
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
