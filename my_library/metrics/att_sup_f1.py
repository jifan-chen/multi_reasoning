from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import get_range_vector, get_device_of
from allennlp.common.checks import ConfigurationError


@Metric.register("att_f1")
class AttF1Measure(Metric):
    """
    Computes Precision, Recall and F1 between predictions and labels.
    The predictions are obtained from thresholding the prediction scores to get a binary results.
    If ``top_k`` is True, assume the prediction_scores contains multiple predictions,
    and choose the one with maximum true positive to compute.
    """
    def __init__(self, positive_th: float, top_k: bool = False) -> None:
        self._positive_th = positive_th
        self._top_k = top_k
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 prediction_scores: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 instance_mask: Optional[torch.Tensor] = None,
                 TH: Optional[float] = None,
                 sum: Optional[bool] = True):
        """
        Parameters
        ----------
        prediction_scores : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...)    if not ``top_k``,
                                             (batch_size, K, ...) if ``top_k``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``attention_scores`` tensor.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        instance_mask: ``torch.Tensor``, optional (default = None).
            A masking tensor os shape (batch_size,).
        TH: ``float``, optional (default = None)
            The threshold to use in current call. If not provided, use ``self._positive_th``.
        """
        prediction_scores, gold_labels, mask, instance_mask = self.unwrap_to_tensors(prediction_scores,
                                                                                    gold_labels,
                                                                                    mask,
                                                                                    instance_mask)
        batch_size = prediction_scores.size(0)
        if TH is None:
            TH = self._positive_th
        
        if instance_mask is None:
            instance_mask = gold_labels.new_ones((gold_labels.size(0),))
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float() * instance_mask.float().view(-1, *([1]*(gold_labels.dim()-1)))
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels
        negative_label_mask = 1.0 - positive_label_mask

        predictions = (prediction_scores >= TH).float()

        # True Positives: correct positively labeled predictions.
        if self._top_k:
            # ``predictions`` gets an extra dimension in axis 1
            # shape : (batch_size, K, ...)
            correct_non_null_predictions = predictions * positive_label_mask.unsqueeze(1)
            masked_correct_non_null_predictions = correct_non_null_predictions * mask.unsqueeze(1)
            # shape : (batch_size, K)
            K_T_P = torch.sum(masked_correct_non_null_predictions,
                              dim=[i for i in range(predictions.dim()) if not i in [0, 1]])
            # shape : (batch_size,)
            T_P, max_idxs = torch.max(K_T_P, dim=1)
            # get the one with max TP within K prediction
            predictions = predictions[get_range_vector(batch_size, get_device_of(mask)), max_idxs]
        else:
            # shape : (batch_size, ...)
            correct_non_null_predictions = predictions * positive_label_mask
            masked_correct_non_null_predictions = correct_non_null_predictions * mask
            # shape : (batch_size,)
            T_P = torch.sum(masked_correct_non_null_predictions,
                            dim=[i for i in range(predictions.dim()) if not i in [0]])
        if not sum:
            self._true_positives += T_P.sum()
        else:
            T_P = T_P.sum()
            self._true_positives += T_P

        if not sum:
            sum_dim = [i for i in range(predictions.dim()) if not i in [0]]

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (1. - predictions) * negative_label_mask
        if not sum:
            T_N = (correct_null_predictions.float() * mask).sum(dim=sum_dim)
            self._true_negatives += T_N.sum()
        else:
            T_N = (correct_null_predictions.float() * mask).sum()
            self._true_negatives += T_N

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (1. - predictions) * positive_label_mask
        if not sum:
            F_N = (incorrect_null_predictions * mask).sum(dim=sum_dim)
            self._false_negatives += F_N.sum()
        else:
            F_N = (incorrect_null_predictions * mask).sum()
            self._false_negatives += F_N

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = predictions * negative_label_mask
        if not sum:
            F_P = (incorrect_non_null_predictions * mask).sum(dim=sum_dim)
            self._false_positives += F_P.sum()
        else:
            F_P = (incorrect_non_null_predictions * mask).sum()
            self._false_positives += F_P

        # T_P, num pred, num label
        return T_P.tolist(), (T_P + F_P).tolist(), (T_P + F_N).tolist()

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
