from typing import Optional

import torch
from torch.nn import functional as F

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_range_vector, get_device_of



## Some Per-Step Reward functions ##
def Evd_Reward(per_step_included, per_step_mask, eos_mask, evd_rclls, evd_f1s,
               strict_eos=False, account_trans=False):
    # If the predict sentence is one of the gold, gets ``f1`` reward.
    # If f1 is 0, the eos (if gets predicted) gets -1, others get 0.
    # If ``strict eos`` is False, eos is always considered gold.
    # Else, eos is considered gold only when the recall is 1.
    # If ``account_trans`` is True, the reward is set as:
    # case 1 - prev: TP and cur: TP -> ``f1``
    # case 2 - prev: TP and cur: FP -> 0
    # case 3 - prev: FP and cur: TP -> 0
    # case 4 - prev: FP and cur: FP -> 0
    batch_size, num_steps = per_step_included.size()
    # Shape: (batch_size,)
    evd_f1s = per_step_included.new_tensor(evd_f1s)
    # Shape: (batch_size, num_steps)
    if account_trans:
        # Since when f1 is 0, all rewards are 0,
        # and when considering eos, only case 1 and 3 can happen,
        # the two processes after ``account_trans`` process won't be affected by the current process.
        prev_per_step_included = F.pad(per_step_included, (1, 0), 'constant', 1)[:, :-1]
        #per_step_rs = prev_per_step_included * (evd_f1s.unsqueeze(1) - (per_step_included == 0).float()) * per_step_mask
        per_step_rs = prev_per_step_included * evd_f1s.unsqueeze(1) * per_step_included * per_step_mask
    else:
        per_step_rs = per_step_included * evd_f1s.unsqueeze(1) * per_step_mask
    # Only give eos reward when recall is 1
    if strict_eos:
        # Shape: (batch_size,)
        evd_rclls = per_step_included.new_tensor(evd_rclls)
        per_step_rs = per_step_rs * (1 - eos_mask) + \
                      ((evd_rclls == 1).float() * evd_f1s).unsqueeze(1) * eos_mask
    '''
    # give the first prediction a -1 reward if ``f1`` is 0
    first_mask = per_step_included.new_zeros((batch_size, num_steps))
    first_mask[:, 0] = 1.
    per_step_rs = torch.where(
                (first_mask == 1) & (evd_f1s == 0).unsqueeze(1),
                torch.tensor(-1.),
                per_step_rs
            )
    '''
    # give the eos a -1 reward if ``f1`` is 0
    per_step_rs = torch.where(
                (eos_mask == 1) & (evd_f1s == 0).unsqueeze(1),
                torch.tensor(-1.),
                per_step_rs
            )
    return per_step_rs


def get_evd_prediction_mask(all_predictions, eos_idx):
    # get the mask w.r.t to ``all_predictions`` that includes the index of the first eos and those before it
    # Shape: (batch_size,)
    batch_size, num_steps = all_predictions.size()
    valid_decoding_lens = torch.sum((all_predictions != eos_idx).float(), dim=1) + 1
    indices = get_range_vector(num_steps, get_device_of(all_predictions)).float()
    mask = (indices.view(1, num_steps) < valid_decoding_lens.view(batch_size, 1)).int()
    eos_mask = (all_predictions == eos_idx).int() * mask
    return mask, eos_mask


@Metric.register("per_step_inclusion")
class PerStepInclusion(Metric):
    """
    Computes Precision, Recall and F1 of the word-to-word lins prediction from attention maps.
    The prediction is computed from attention scores wih respect to the given positive threshold.
    """
    def __init__(self, eos_idx: int) -> None:
        self._eos_idx = eos_idx
        self._tot_predictions = 0.0
        self._acc_predictions = 0.0

    def __call__(self,
                 all_predictions: torch.Tensor,
                 sent_labels: torch.Tensor,
                 sent_mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        all_predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, num_decoding_steps). The prediction could
            equal to ``eos_idx``
        sent_labels : ``torch.Tensor``, required.
            A tensor of one-hot label of shape (batch_size, num_sents) that specifies which sentences
            are marked as evidences.
        sent_mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``sent_labels``.
        """
        all_predictions, sent_labels, sent_mask = self.unwrap_to_tensors(all_predictions, sent_labels, sent_mask)
        
        if sent_mask is None:
            sent_mask = torch.ones_like(sent_labels)
        sent_mask = sent_mask.float()
        sent_labels = sent_labels.float()
        all_predictions = all_predictions.float()

        # prepend the labels with one to represent that the end symbol is also an accurate prediction
        # Shape: (batch_size, 1+num_sents)
        sent_labels = F.pad(sent_labels, (1, 0), 'constant', 1)

        # Also prepend the sent mask with one
        # Shape: (batch_size, 1+num_sents)
        sent_mask = F.pad(sent_mask, (1, 0), 'constant', 1)

        batch_size, num_sents = sent_labels.size()
        num_steps = all_predictions.size(1)
        # Transform the predicted sent indices to one-hot vector in the same form with sent_labels
        indices = get_range_vector(num_sents, get_device_of(sent_labels)).float()
        # shaps: (batch_size, num_steps, num_sents)
        preds_onehot = (all_predictions.view(batch_size, num_steps, 1) == indices.view(1, 1, num_sents)).float()

        # Check whether each predicted evidence is either one of the gold evidence or the end symbol
        # The values in ``per_step_included`` should all be 0 or 1 since each vector in ``preds_onehot``
        # is one-hot, and at the position where ``sent_labels`` is -1, the value of ``pred_onehot`` should be 0
        # Shape: (batch_size, num_steps)
        per_step_included = torch.sum(preds_onehot * sent_labels.view(batch_size, 1, num_sents), dim=2)

        # record the number of predictions, and the number of correct predictions,
        # which is actually the precision
        per_step_mask, eos_mask = get_evd_prediction_mask(all_predictions, self._eos_idx)
        per_step_mask = per_step_mask.float()
        eos_mask = eos_mask.float()
        self._tot_predictions += torch.sum(per_step_mask)
        self._acc_predictions += torch.sum(per_step_included * per_step_mask)
        
        return per_step_included, per_step_mask, eos_mask

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        """
        precision = float(self._acc_predictions) / float(self._tot_predictions + 1e-13)
        if reset:
            self.reset()
        return precision

    def reset(self):
        self._tot_predictions = 0.0
        self._acc_predictions = 0.0
