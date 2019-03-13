from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("att_f1")
class AttF1Measure(Metric):
    """
    Computes Precision, Recall and F1 of the word-to-word lins prediction from attention maps.
    The prediction is computed from attention scores wih respect to the given positive threshold.
    """
    def __init__(self, positive_th: float) -> None:
        self._positive_th = positive_th
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 attention_scores: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 TH: Optional[float] = None):
        """
        Parameters
        ----------
        attention_scores : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``attention_scores`` tensor.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        TH: ``float``, optional (default = None)
            The threshold to use in current call. If not provided, use ``self._positive_th``.
        """
        attention_scores, gold_labels, mask = self.unwrap_to_tensors(attention_scores, gold_labels, mask)
        if TH is None:
            TH = self._positive_th
        
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels
        negative_label_mask = 1.0 - positive_label_mask

        predictions = (attention_scores >= TH).float()

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (1. - predictions) * negative_label_mask
        T_N = (correct_null_predictions.float() * mask).sum()
        self._true_negatives += T_N

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = predictions * positive_label_mask
        T_P = (correct_non_null_predictions * mask).sum()
        self._true_positives += T_P

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (1. - predictions) * positive_label_mask
        F_N = (incorrect_null_predictions * mask).sum()
        self._false_negatives += F_N

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = predictions * negative_label_mask
        F_P = (incorrect_non_null_predictions * mask).sum()
        self._false_positives += F_P

        # T_P, num pred, num label
        return T_P.item(), (T_P + F_P).item(), (T_P + F_N).item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
