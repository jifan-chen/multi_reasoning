from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import torch
import scipy.stats as ss
from scipy.stats import kendalltau
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, Average, CategoricalAccuracy
from my_library.metrics import AttF1Measure
from my_library.metrics import RankCorrelation


@Model.register("hotpot_bert_reranker")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        encoding_dim = text_field_embedder.get_output_dim()

        self._dropout = torch.nn.Dropout(p=dropout)

        self._mse = nn.MSELoss()

        self._categorical_acc = CategoricalAccuracy()

        self._coref_f1_metric = AttF1Measure(0.1)

        self.linear_start = nn.Linear(encoding_dim, 1)

        self.linear_end = nn.Linear(encoding_dim, 1)

        self.linear_type = nn.Linear(encoding_dim, 2)

        self._loss_trackers = {'loss': Average()}

        self._sample_count = 0

        self._predicted_rank = []

        self._gold_rank = []

        self._accuracy = BooleanAccuracy()

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor] = None,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        question_passage_bert = self._text_field_embedder(question_passage)
        qp_cls_rep = torch.index_select(question_passage_bert, 1, torch.tensor([0]).cuda())

        qp_score = self.linear_type(qp_cls_rep)

        batch_size = qp_score.size()[0]
        predict = torch.argmax(qp_score, dim=-1)
        predict_loss = nll_loss(util.masked_log_softmax(qp_score.squeeze(1), None), label)
        loss = predict_loss
        self._loss_trackers['loss'](loss)
        output_dict = {"loss": loss,
                       "score": qp_score,
                       "predict": predict,
                       "label": label,
                       "_id": metadata[0]['id']}

        self._accuracy(predict, label)
        if metadata is not None:
            for i in range(batch_size):
                pass

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        acc = self._accuracy.get_metric(reset)
        metrics = {
                'accuracy': acc
                }
        # for name, tracker in self._loss_trackers.items():
        #     metrics[name] = tracker.get_metric(reset)
        return metrics


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
