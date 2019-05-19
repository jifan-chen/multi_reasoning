from typing import Optional
import torch
from torch.autograd import Variable
from allennlp.training.metrics.metric import Metric
from scipy.stats import kendalltau
import scipy.stats as ss


@Metric.register("rank_correlation")
class RankCorrelation(Metric):
    """
    Computes Precision, Recall and F1 of the word-to-word lins prediction from attention maps.
    The prediction is computed from attention scores wih respect to the given positive threshold.
    """

    def __init__(self) -> None:
        self._tau_count = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predict_rank,
                 gold_rank,
                 mask: Optional[torch.Tensor] = None):

        gold_rank = ss.rankdata(gold_rank, method='ordinal')
        predict_rank = ss.rankdata(predict_rank, method='ordinal')
        tau, p_value = kendalltau(gold_rank, predict_rank)

        self._tau_count += tau
        self._total_count += 1

    def get_metric(self, reset: bool = False):
        tau = float(self._tau_count) / float(self._total_count + 1e-20)
        if reset:
            self.reset()
        return tau

    def reset(self):
        self._tau_count = 0.0
        self._total_count = 0.0
