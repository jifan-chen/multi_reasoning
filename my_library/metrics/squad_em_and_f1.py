from typing import Tuple

from overrides import overrides

from allennlp.tools import squad_eval
from allennlp.training.metrics import Metric, SquadEmAndF1


@Metric.register("squad_rt")
class SquadEmAndF1_RT(SquadEmAndF1):
    """
    A modified version of SquadEmAndF1 that returns the em and f1 score per call.
    """
    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        exact_match = squad_eval.metric_max_over_ground_truths(
                squad_eval.exact_match_score,
                best_span_string,
                answer_strings)
        f1_score = squad_eval.metric_max_over_ground_truths(
                squad_eval.f1_score,
                best_span_string,
                answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1
        return exact_match, f1_score
