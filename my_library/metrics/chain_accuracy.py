from typing import Optional

from overrides import overrides
import torch
from torch.nn import functional as F

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import get_range_vector, get_device_of


@Metric.register("chain_accuracy")
class ChainAccuracy(Metric):
    """
    Checks the correctness of evdience prediction for each step in a given chain with repsect to the label.
    It may happen that the length of the predicted chain is shorter than that of the labeled chain (like during validation).
    In this case, the predicted chain or the labeled chain will be padded zero,
    that is, the eos idx, so that the lengths of two tensors are equal.
    If the provided predictions contains multiple chains (such as in beamsearch), takes the chain with maximum true positive.
    """
    def __init__(self) -> None:
        self._correct_count = 0.
        self._total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 predict_mask: Optional[torch.Tensor] = None,
                 gold_mask: Optional[torch.Tensor] = None,
                 instance_mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, K, decoding_step).
        gold_labels : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, true_decoding_step).
        predict_mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        gold_mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``gold_labels``.
        instance_mask: ``torch.Tensor``, optional (default = None).
            A tensor of shape (batch_size,).
        """
        predictions, gold_labels, predict_mask, gold_mask, instance_mask = self.unwrap_to_tensors(predictions,
                                                                                                  gold_labels,
                                                                                                  predict_mask,
                                                                                                  gold_mask,
                                                                                                  instance_mask)
        batch_size = predictions.size(0)
        predictions = predictions.float()
        gold_labels = gold_labels.float()
        predict_mask = predict_mask.float()
        gold_mask = gold_mask.float()
        instance_mask = instance_mask.float()
        if instance_mask is None:
            instance_mask = gold_labels.new_ones((batch_size,)).float()

        # pad predictions or gold labels so that they have equal length
        predict_decoding_step = predictions.size(2)
        true_decoding_step = gold_labels.size(1)
        if predict_decoding_step < true_decoding_step:
            # shape: (batch_size, K, true_decoding_step)
            predictions = F.pad(predictions, (0, true_decoding_step - predict_decoding_step), 'constant', 0)
            # shape: (batch_size, K, true_decoding_step)
            predict_mask = F.pad(predict_mask, (0, true_decoding_step - predict_decoding_step), 'constant', 0)
        elif predict_decoding_step > true_decoding_step:
            # shape: (batch_size, predict_decoding_step)
            gold_labels = F.pad(gold_labels, (0, predict_decoding_step - true_decoding_step), 'constant', 0)
            # shape: (batch_size, predict_decoding_step)
            gold_mask = F.pad(gold_mask, (0, predict_decoding_step - true_decoding_step), 'constant', 0)
        # OR the mask so that all valid symbols in the label and prediction will contribute to accuracy
        mask = (predict_mask.byte() | gold_mask.unsqueeze(1).byte()).float()

        # shape: (batch_size, K, max(predict_decoding_step, true_decoding_step))
        K_correct = (predictions == gold_labels.unsqueeze(1)).float() * mask

        # shape: (batch, K)
        K_correct_num = torch.sum(K_correct, dim=-1)
        K_all_num = torch.sum(mask, dim=-1)

        # shape: (batch_size,)
        max_correct_num, max_correct_idx = torch.max(K_correct_num, dim=-1)
        max_all_num = K_all_num[get_range_vector(batch_size, get_device_of(mask)), max_correct_idx]

        print("max_correct_num", max_correct_num)
        print("max_all_num", max_all_num)
        print("instance_mask", instance_mask)

        self._correct_count += (max_correct_num * instance_mask).sum()
        self._total_count += (max_all_num * instance_mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self._correct_count) / float(self._total_count + 1e-13)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
