from typing import Optional

import torch
from torch.autograd import Variable
from allennlp.training.metrics.metric import Metric


@Metric.register("sent_acc")
class SentAcc(Metric):
    """
    Computes Precision, Recall and F1 of the word-to-word lins prediction from attention maps.
    The prediction is computed from attention scores wih respect to the given positive threshold.
    """

    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predict_labels: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):

        batch_size = predict_labels.size(0)
        predict_labels = predict_labels.view(batch_size)
        gold_labels = gold_labels.view(batch_size, -1)
        mask = (gold_labels >= 0).long()
        # print('predict:', predict_labels)
        # print('gold_labels:', gold_labels)
        predict_onehot = self.to_one_hot(predict_labels, gold_labels.size(1))
        # print('overlap:', predict_onehot)
        predict = torch.sum(predict_onehot.float().cuda() * gold_labels.float(), dim=-1)
        predict = (predict >= 1).long()

        correct = torch.sum(predict).item()
        self._correct_count += correct
        self._total_count += batch_size

    @staticmethod
    def to_one_hot(y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

    def get_metric(self, reset: bool = False):
        accuracy = float(self._correct_count) / float(self._total_count)
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
