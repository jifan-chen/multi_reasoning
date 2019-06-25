from typing import Dict, Tuple, List, Any, Union, Callable
import warnings

import math
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, GRUCell
from torch import nn
from torch.nn import functional as F

from allennlp.modules import Attention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name



class DecodeHelper_:
    def __init__(self,
                 eos_idx: int,
                 max_steps: int):
        self._eos_idx = eos_idx
        self.max_steps = max_steps

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given the logits for each class, return the predicted class and its logit.
        Parameters
        ----------
        logits : ``torch.Tensor``
            The tensor contains the logits for each class.
            Shape: ``(batch_size, class_nums)``
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(logprobs, predicted_classes)``, where ``logprobs``
            has shape ``(batch_size,)`` and ``predicted_classes``
            has shape ``(batch_size,)``.
        """
        raise NotImplementedError

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply greedy search to find the
        target sequences.
        Parameters
        ----------
        start_embedding : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``StateType``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept three arguments. The first being the current state.
            The second being a tensor of shape ``(batch_size,)``, representing the index of the predicted
            tokens from the last time step. The third being a tensor of shape 
            ``(batch_size, decoder_input_dim)``, representing the give next input. If the second is provided,
            The decoder will predict the next output by previous prediction. Otherwise, the third sould be
            provided and used for predicting the next output.
            The function is expected to return a tuple, where the first element
            is the updated state. The tensor in the state should have shape. The second element
            is a tensor of shape ``(batch_size, class_num)`` containing
            the log probabilities of the tokens for the next step, ``(batch_size, *)``,
            where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of ``(all_predictions, all_logprobs, last_log_probabilities)``, where ``all_predictions``
            and ``all_logprobs`` has shape ``(batch_size, max_steps)``, and ``log_probabilities``
            has shape ``(batch_size,)``.
        """
        raise NotImplementedError


class TeacherForcingHelper(DecodeHelper_):
    def _sample(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Since we got labels, we also return the log probs of the labels
        """
        # shape(labels): (batch_size,)
        preds_logprobs, preds = torch.max(logits, dim=1)
        return preds_logprobs, preds, torch.gather(logits, 1, labels[:, None]).squeeze(1)

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = start_embedding.size()[0]
        labels = labels.long()

        # List of (batch_size,) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size,) tensors. One for each time step. The log
        # probability corresponding to the true labels.
        label_logprobs: List[torch.Tensor] = []

	# Calculate the first timestep. This is done outside the main loop
        # so that we can do some initialization for the loop.
        # shape: (batch_size, num_classes)
        state, start_class_log_probabilities = step(start_state, None, start_embedding)

        num_classes = start_class_log_probabilities.size()[1]

        # shape: (batch_size,), (batch_size,), (batch_size,)
        start_log_probabilities, \
        start_predicted_classes, \
        label_start_log_probabilities = self._sample(start_class_log_probabilities, labels[:, 0])

        # The log probabilities for the last time step. Here we use the logprob of predicting labels
        # shape: (batch_size,)
        label_last_log_probabilities = label_start_log_probabilities

        # shape: [(batch_size,)]
        predictions.append(start_predicted_classes)
        label_logprobs.append(label_start_log_probabilities)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._eos_idx] = 0.

        for timestep in range(labels.size(1) - 1):
            # shape: (batch_size,)
            labels_tm1 = labels[:, timestep]

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size, num_classes)
            state, class_log_probabilities = step(state, labels_tm1, None)

            # shape: (batch_size, num_classes)
            labels_tm1_expanded = labels_tm1.unsqueeze(-1).expand(
                    batch_size,
                    num_classes
            )

            # Here we are finding any sequences that alreadly end in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution to let the log_probabilities fall through.
            # shape: (batch_size, num_classes)
            cleaned_log_probabilities = torch.where(
                    labels_tm1_expanded == self._eos_idx,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # Find the logprbs w.r.t the labels for this timestep.
            # shape: (batch_size,), (batch_size,), (batch_size,)
            log_probabilities, \
            predicted_classes, \
            label_log_probabilities = self._sample(cleaned_log_probabilities, labels[:, timestep+1])

            predictions.append(predicted_classes)
            label_logprobs.append(label_log_probabilities)

            # shape: (batch_size, num_classes)
            label_last_log_probabilities = label_last_log_probabilities + label_log_probabilities

        if not torch.isfinite(label_last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # shape: (batch_size, max_steps)
        all_predictions = torch.stack(predictions, dim=1)
        all_label_logprobs = torch.stack(label_logprobs, dim=1)
        assert torch.allclose(torch.sum(all_label_logprobs, dim=-1), label_last_log_probabilities)

        return all_predictions, all_label_logprobs, label_last_log_probabilities, state["cell_hidden"]


class GreedyHelper(DecodeHelper_):
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.max(logits, dim=1)

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = start_embedding.size()[0]

        # List of (batch_size,) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size,) tensors. One for each time step. The log
        # probability corresponding to the prediction.
        logprobs: List[torch.Tensor] = []

	# Calculate the first timestep. This is done outside the main loop
        # so that we can do some initialization for the loop.
        # shape: (batch_size, num_classes)
        state, start_class_log_probabilities = step(start_state, None, start_embedding)

        num_classes = start_class_log_probabilities.size()[1]

        # shape: (batch_size,), (batch_size,)
        start_log_probabilities, start_predicted_classes = self._sample(start_class_log_probabilities)
        if (start_predicted_classes == self._eos_idx).all():
            warnings.warn("Empty sequences predicted. You may want to "
                          "ensure your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), \
                   start_log_probabilities.unsqueeze(-1), \
                   start_log_probabilities, \
                   state["cell_hidden"]

        # The log probabilities for the last time step.
        # shape: (batch_size,)
        last_log_probabilities = start_log_probabilities

        # shape: [(batch_size,)]
        predictions.append(start_predicted_classes)
        logprobs.append(start_log_probabilities)
        state['hist_predictions'] = start_predicted_classes.reshape(batch_size, 1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._eos_idx] = 0.

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size,)
            last_predictions = predictions[-1]

            # If every predicted token from the last step is `self._eos_idx`,
            # then we can stop early.
            if (last_predictions == self._eos_idx).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size, num_classes)
            state, class_log_probabilities = step(state, last_predictions, None)

            # shape: (batch_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size,
                    num_classes
            )

            # Here we are finding any sequences where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the sequence to predict the end token
            # this timestep as well and also let the log_probabilities fall through.
            # shape: (batch_size, num_classes)
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._eos_idx,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # Find the predicted class for this timestep greedily.
            # shape: (batch_size,), (batch_size,)
            log_probabilities, predicted_classes = self._sample(cleaned_log_probabilities)

            predictions.append(predicted_classes)
            logprobs.append(log_probabilities)
            state['hist_predictions'] = torch.cat([state['hist_predictions'],
                                                   predicted_classes.reshape(batch_size, 1)],
                                                  dim=1)

            # shape: (batch_size, num_classes)
            last_log_probabilities = last_log_probabilities + log_probabilities

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # shape: (batch_size, max_steps)
        all_predictions = torch.stack(predictions, dim=1)
        assert (all_predictions == state['hist_predictions']).all()
        all_logprobs = torch.stack(logprobs, dim=1)
        assert torch.allclose(torch.sum(all_logprobs, dim=-1), last_log_probabilities)

        return all_predictions, all_logprobs, last_log_probabilities, state["cell_hidden"]


class SamplingHelper(GreedyHelper):
    def _sample(self, logits: torch.Tensor, temp=0.5) -> torch.Tensor:
        # shape: (batch_size, 1)
        samples = torch.multinomial(torch.exp(logits / temp), 1)
        #print("sample:", samples)
        # shape: (batch_size, 1)
        logprobs = torch.gather(logits, 1, samples)
        #print("sample logprobs:", logprobs)
        # shape: (batch_size,), (batch_size,)
        return logprobs.squeeze(1), samples.squeeze(1)


class BeamSearchHelper(DecodeHelper_):
    def __init__(self, beam_size, **kwargs):
        super(BeamSearchHelper, self).__init__(**kwargs)
        self.beam_size = beam_size

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        The logits here will possibly have shape ``(batch_size, beam_size*class_nums)``
        """
        return logits.topk(self.beam_size)

    def _sample_t0(self, logits: torch.Tensor) -> torch.Tensor:
        """
        The logits here will possibly have shape ``(batch_size, beam_size*class_nums)``
        """
        batch_size, bxc = logits.size()
        if bxc >= self.beam_size:
            return self._sample(logits)
        else:
            # need to take into account the case when the number of sentences is less then beam size QQ
            # this will only happens at the firt time step, since once it is considered at t = 0, then
            # all the following logits will have beam_size*class_nums (>= beam_size) in dim 1
            # shape: (batch_size, bxc)
            expand_times = math.ceil(self.beam_size / bxc)
            top_logits, top_idxs = logits.topk(bxc)
            # shape: (batch_size, beam_size)
            top_logits = top_logits[:, :, None].\
                    expand(batch_size, bxc, expand_times).\
                    reshape(batch_size, bxc*expand_times)[:, :self.beam_size]
            top_idxs = top_idxs[:, :, None].\
                    expand(batch_size, bxc, expand_times).\
                    reshape(batch_size, bxc*expand_times)[:, :self.beam_size]
            return top_logits, top_idxs

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Notes
            The ``batch_size`` in the input and output of the step function 
            will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
        """
        batch_size = start_embedding.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. The log
        # probability corresponding to the prediction.
        logprobs: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

	# Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        state, start_class_log_probabilities = step(start_state, None, start_embedding)

        num_classes = start_class_log_probabilities.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = self._sample_t0(start_class_log_probabilities)
        if self.beam_size == 1 and (start_predicted_classes == self._eos_idx).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), \
                   start_top_log_probabilities.unsqueeze(-1), \
                   start_top_log_probabilities, \
                   state["cell_hidden"].\
                        unsqueeze(1).\
                        expand(batch_size, self.beam_size, state["cell_hidden"].size(1)).\
                        reshape(batch_size * self.beam_size, state["cell_hidden"].size(1))

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)
        logprobs.append(start_top_log_probabilities)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size * self.beam_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._eos_idx] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                    unsqueeze(1).\
                    expand(batch_size, self.beam_size, *last_dims).\
                    reshape(batch_size * self.beam_size, *last_dims)
        state['hist_predictions'] = start_predicted_classes.reshape(batch_size * self.beam_size, 1)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._eos_idx`,
            # then we can stop early.
            if (last_predictions == self._eos_idx).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            state, class_log_probabilities = step(state, last_predictions, None)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size * self.beam_size,
                    num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well and also let the log_probabilities fall through.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._eos_idx,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # shape: (batch_size * beam_size, 1)
            last_log_probabilities = last_log_probabilities.reshape(batch_size * self.beam_size, 1)
            # shape: (batch_size, beam_size * num_classes)
            last_log_probabilities = (last_log_probabilities + cleaned_log_probabilities).view(
                    batch_size, self.beam_size * num_classes
            )
            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            last_log_probabilities, topk_idxs = self._sample(last_log_probabilities)
            # shape: (batch_size, beam_size)
            tok_idxs_clean_logprobs = cleaned_log_probabilities.view(
                    batch_size, self.beam_size * num_classes
            ).gather(1, topk_idxs)

            # The beam indices come from a `beam_size * num_classes` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by num_classes gives the ancestor.
            # shape: (batch_size, beam_size)
            backpointer = topk_idxs // num_classes
            backpointers.append(backpointer)

            # The position of an index inside the group of its ancestor indicates the predicted class. Hence
            # take the modulus
            # shape: (batch_size, beam_size)
            predicted_classes = topk_idxs % num_classes
            predictions.append(predicted_classes)
            logprobs.append(tok_idxs_clean_logprobs)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                        view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                        expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                        reshape(batch_size, self.beam_size, *last_dims).\
                        gather(1, expanded_backpointer).\
                        reshape(batch_size * self.beam_size, *last_dims)
            state['hist_predictions'] = torch.cat([state['hist_predictions'],
                                                   predicted_classes.reshape(batch_size * self.beam_size, 1)],
                                                  dim=1)

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        reconstructed_logprobs = [logprobs[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            cur_logprobs = logprobs[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)
            reconstructed_logprobs.append(cur_logprobs)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        final_logprobs = logprobs[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)
        reconstructed_logprobs.append(final_logprobs)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_logprobs = torch.cat(list(reversed(reconstructed_logprobs)), 2)

        assert (all_predictions == state['hist_predictions'].view(batch_size, self.beam_size, -1)).all()
        assert torch.allclose(torch.sum(all_logprobs, dim=-1), last_log_probabilities)

        final_hidden = state["cell_hidden"].reshape(batch_size, self.beam_size, -1)

        return all_predictions, all_logprobs, last_log_probabilities, final_hidden


class PointerNetDecoder(torch.nn.Module):
    """
    Most of the codes are coming from the decoding part of CopyNetSeq2Seq from AllenNLP
    Parameters
    ----------
    attention : ``MatrixAttention``, required
        This is used to get the alignment scores between decoder hidden state and the memory.
    memory_dim : ``int``, required
        The embedding dimension of the encoder outputs (memory).
    aux_input_dim : ``int``, optional
        The embedding dimension of the auxilary decoder input.
    beam_size : ``int``, optional (default: 3)
        Beam width to use for beam search prediction.
    max_decoding_steps : ``int``, optional (default: 5)
        Maximum sequence length of target predictions.
    """

    def __init__(self,
                 attention: MatrixAttention,
                 memory_dim: int,
                 aux_input_dim: int = None,
                 train_helper='sample',
                 val_helper='beamsearch',
                 beam_size: int = 3,
                 max_decoding_steps: int = 5,
                 predict_eos: bool = True,
                 cell: str = 'lstm') -> None:
        super(PointerNetDecoder, self).__init__()                
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = memory_dim
        self.decoder_output_dim = memory_dim
        self.decoder_input_dim = memory_dim
        self.aux_input_dim = aux_input_dim or 0

        # The decoder input will be the embedding vector that the decoder is pointing to 
        # in the memory in the previous timestep concatenated with some auxiliary input if provided.
        self._attention = attention
        self._input_projection_layer = Linear(
                self.encoder_output_dim + self.aux_input_dim,
                self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        self._cell = cell
        if cell == 'lstm':
            self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)
        elif cell == 'gru':
            self._decoder_cell = GRUCell(self.decoder_input_dim, self.decoder_output_dim)

        # At initial step we take the trainable ``start embedding`` as input
        self._start_embedding = nn.Parameter(torch.Tensor(1, self.decoder_input_dim))
        nn.init.normal_(self._start_embedding, std=0.01)

        # The end embedding which we prepend to the memmory so that the pointernet
        # can learn to terminate the prediction if necessary
        self._end_embedding = nn.Parameter(torch.Tensor(1, self.encoder_output_dim), requires_grad=True)
        nn.init.normal_(self._end_embedding, std=0.01)

        # We set the eos index to zero since the end embedding is prepended to the memory
        self._eos_idx = 0
        self._predict_eos = predict_eos

        def get_helper(helper_type):
            if helper_type == 'teacher_forcing':
                return TeacherForcingHelper(eos_idx=self._eos_idx,
                                            max_steps=max_decoding_steps)
            elif helper_type == 'sample':
                return SamplingHelper(eos_idx=self._eos_idx,
                                      max_steps=max_decoding_steps)
            elif helper_type == 'greedy':
                return GreedyHelper(eos_idx=self._eos_idx,
                                    max_steps=max_decoding_steps)
            elif helper_type == 'beamsearch':
                return BeamSearchHelper(beam_size=beam_size,
                                        eos_idx=self._eos_idx,
                                        max_steps=max_decoding_steps)
            else:
                raise ValueError("Unsupported Decoding Helper!")
        # Training decoding helper
        self._train_helper_type = train_helper
        self._train_decoding_helper = get_helper(train_helper)
        # At prediction time, we'll use a beam search to find the best target sequence.
        self._eval_helper_type = val_helper
        self._eval_decoding_helper = get_helper(val_helper)

    @overrides
    def forward(self,  # type: ignore
                memory: torch.Tensor,
                memory_mask: torch.Tensor,
                init_hidden_state: torch.Tensor = None,
                aux_input: torch.Tensor = None,
                transition_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        memory : ``torch.FloatTensor``, required
            The output of an encoder. Shape: ``[batch_size, memory_len, encoder_output_dim]``
        memory_mask : ``torch.FloatTensor``, required
            The mask of the encoder output. Shape: ``[batch_size, memory_len]``
        init_hidden_state : ``torch.FloatTensor``, optional
            The initial hidden state for the decoder cell. Shape: ``[batch_size, decoder_output_dim]``.
            If not provided, the hidden state will be initialized with zero.
        aux_input : ``torch.FloatTensor``, optional
            The auxilary information that will be feed to the deocder in every step.
            Shape: ``[batch_size, aux_input_dim]``.
        transition_mask : ``torch.FloatTensor``, optional
            The mask for restricting the action space. Shape: ``[batch_size, memory_len, memory_len]``
        labels : ``torch.IntTensor``, optional
            The true labels for teacher forcing. Shape: ``[batch_size, decoding_len]``
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch_size = memory.size(0)
        start_state = self._init_decoder_state(memory, memory_mask, init_hidden_state, aux_input, transition_mask)
        # shape: (batch_size, decoder_input_dim)
        start_embedding = self._start_embedding.expand(batch_size, self.decoder_input_dim)
        if self.training:
            helper = self._train_decoding_helper
        else:
            helper = self._eval_decoding_helper
        # shape (all_predictions): (batch_size, K, num_decoding_steps)
        # shape (all_logprobs): (batch_size, K, num_decoding_steps)
        # shape (seq_logprobs): (batch_size, K)
        # shape (final_hidden): (batch_size, K, decoder_output_dim)
        all_predictions, all_logprobs, seq_logprobs, final_hidden = helper.search(
                start_embedding, start_state, self._decoder_step, labels)
        # add a extra beam dimension if needed
        if all_predictions.dim() == 2:
            all_predictions = all_predictions.unsqueeze(1)
            all_logprobs = all_logprobs.unsqueeze(1)
            seq_logprobs = seq_logprobs.unsqueeze(1)
            final_hidden = final_hidden.unsqueeze(1)
        return all_predictions, all_logprobs, seq_logprobs, final_hidden

    def _init_decoder_state(self,
                            memory: torch.Tensor,
                            memory_mask: torch.Tensor,
                            init_hidden_state: torch.Tensor = None,
                            aux_input: torch.Tensor = None,
                            transition_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Initialize the state to be passed to the decoder at the first decoding time step.
        """
        batch_size, memory_len = memory_mask.size()
        # Prepend the end embedding to the memory so that the model
        # can terminate the decoding by attend on the end vector
        # shape: (batch_size, 1, encoder_output_dim)
        # shape: (batch_size, 1+memory_len, encoder_output_dim)
        prepend = self._end_embedding.expand(batch_size, self.encoder_output_dim).unsqueeze(1)
        memory = torch.cat([prepend, memory], dim=1)
        # The memory mask also need to be prepended
        # If ``predict_eos`` prepend ones, else zeros
        # shape: (batch_size, 1)
        # shape: (batch_size, 1+memory_len)
        if self._predict_eos:
            mask_prepend = memory_mask.new_ones((batch_size, 1))
        else:
            mask_prepend = memory_mask.new_zeros((batch_size, 1))
        memory_mask = torch.cat([mask_prepend, memory_mask], dim=1)

        # Initialize the decoder hidden state with zeros if not provided.
        # shape: (batch_size, decoder_output_dim)
        if init_hidden_state is None:
            init_hidden_state = memory.new_zeros(batch_size, self.decoder_output_dim)

        state = {"memory":      memory,
                 "memory_mask": memory_mask,
                 "cell_hidden": init_hidden_state}

        if self._cell == 'lstm':
            # Initialize the decoder context with zeros. Only do it when the cell type is ``LSTMCell``
            # shape: (batch_size, decoder_output_dim)
            init_context = memory.new_zeros(batch_size, self.decoder_output_dim)
            state["cell_context"] = init_context

        if not aux_input is None:
            state['aux_input'] = aux_input

        # If transition mask is provided, we also need to prepend ones on the row and column axis
        # to account for the eos.
        if not transition_mask is None:
            # shape: (batch_size, 1+memory_len, 1+memory_len)
            transition_mask = F.pad(transition_mask, (1, 0, 1, 0), 'constant', int(self._predict_eos))
            state['transition_mask'] = transition_mask

        return state

    def _decoder_step(self,
                      state: Dict[str, torch.Tensor],
                      last_predictions: torch.Tensor = None,
                      next_input: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # shape: (group_size, memory_len, encoder_output_dim)
        memory = state["memory"]
        # shape: (group_size, memory_len)
        memory_mask = state["memory_mask"].float()
        group_size, memory_len, _ = memory.size()

        if not last_predictions is None:
            # We obtain the next input by last predictions
            # The last predictions represent the indices pointing to encoder outputs.
            # We gather the encoder outputs w.r.t the last predictions and take them as the next inputs
            # shape: (group_size, 1, encoder_output_dim)
            expand_last_predictions = last_predictions[:, None, None].expand(group_size, 1, self.encoder_output_dim)
            # shape: (group_size, encoder_output_dim)
            decoder_input = torch.gather(memory, 1, expand_last_predictions).squeeze(1)

            # We also gather the transition restriction, if provided, w.r.t the last predictions,
            # and apply on the memory mask.
            # shape: (group_size, memory_len)
            if not state.get("transition_mask", None) is None:
                # shape: (group_size, 1, memory_len)
                expand_last_predictions = last_predictions[:, None, None].expand(group_size, 1, memory_len)
                transition_mask = torch.gather(state["transition_mask"], 1, expand_last_predictions).squeeze(1)
                memory_mask = memory_mask * transition_mask.float()
        else:
            # We use ``next input`` as the next decoder input. This condition is use for the initial step.
            assert not next_input is None
            # shape: (group_size, decoder_input_dim)
            decoder_input = next_input

        # We apply another mask to prevent the model repeatedly selecting the same memory to point to.
        if "hist_predictions" in state:
            hist_attention = memory.new_zeros((group_size, memory_len))
            hist_attention.scatter_(1, state["hist_predictions"], 1.)
            memory_mask = memory_mask * (1. - hist_attention)

        # shape: (group_size, encoder_output_dim + aux_input_dim)
        if not state.get('aux_input', None) is None:
            decoder_input = torch.cat((decoder_input, state['aux_input']), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        if self._cell == 'lstm':
            # shape: (group_size, decoder_output_dim), (group_size, decoder_output_dim)
            next_cell_hidden, next_cell_context = self._decoder_cell(
                    projected_decoder_input,
                    (state["cell_hidden"], state["cell_context"]))
        elif self._cell == 'gru':
            # shape: (group_size, decoder_output_dim)
            next_cell_hidden = self._decoder_cell(
                    projected_decoder_input,
                    state["cell_hidden"])
        if not last_predictions is None:
            # Here we are finding any sequences where we predicted the end token in
            # the previous timestep and directly passing through its hidden state.
            # shape: (group_size, decoder_output_dim)
            expand_last_predictions = last_predictions.unsqueeze(-1).expand(
                    group_size,
                    self.decoder_output_dim
            )
            # shape: (group_size, decoder_output_dim)
            state["cell_hidden"] = torch.where(
                    expand_last_predictions == self._eos_idx,
                    state["cell_hidden"],
                    next_cell_hidden
            )
            if self._cell == 'lstm':
                # shape: (group_size, decoder_output_dim)
                state["cell_context"] = torch.where(
                        expand_last_predictions == self._eos_idx,
                        state["cell_context"],
                        next_cell_context
                )

        # Compute the alignments scores w.r.t the encoder outputs
        # shape: (group_size, 1, decoder_output_dim)
        query = state["cell_hidden"].unsqueeze(1)
        # shape: (group_size, memory_len)
        attention_matrix = self._attention(query, memory).squeeze(1)
        attention_logprobs = util.masked_log_softmax(attention_matrix, memory_mask)
        #attention_probs = util.masked_softmax(attention_matrix, memory_mask)
        return state, attention_logprobs#, attention_probs


class AdpMemPointerNetDecoder(torch.nn.Module):
    """
    Most of the codes are coming from the decoding part of CopyNetSeq2Seq from AllenNLP
    Parameters
    ----------
    attention : ``MatrixAttention``, required
        This is used to get the alignment scores between decoder hidden state and the memory.
    memory_dim : ``int``, required
        The embedding dimension of the encoder outputs (memory).
    aux_input_dim : ``int``, optional
        The embedding dimension of the auxilary decoder input.
    beam_size : ``int``, optional (default: 3)
        Beam width to use for beam search prediction.
    max_decoding_steps : ``int``, optional (default: 5)
        Maximum sequence length of target predictions.
    """

    def __init__(self,
                 attention: MatrixAttention,
                 fg_attention: MatrixAttention,
                 memory_dim: int,
                 aux_input_dim: int = None,
                 train_helper='sample',
                 val_helper='beamsearch',
                 beam_size: int = 3,
                 max_decoding_steps: int = 5,
                 predict_eos: bool = True,
                 cell: str = 'lstm') -> None:
        super(AdpMemPointerNetDecoder, self).__init__()                
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = memory_dim
        self.decoder_output_dim = memory_dim
        self.decoder_input_dim = memory_dim
        self.aux_input_dim = aux_input_dim or 0

        # The decoder input will be the embedding vector that the decoder is pointing to 
        # in the memory in the previous timestep concatenated with some auxiliary input if provided.
        self._attention = attention
        self._fg_attention = fg_attention
        self._input_projection_layer = Linear(
                self.encoder_output_dim + self.aux_input_dim,
                self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        self._cell = cell
        if cell == 'lstm':
            self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)
        elif cell == 'gru':
            self._decoder_cell = GRUCell(self.decoder_input_dim, self.decoder_output_dim)

        # At initial step we take the trainable ``start embedding`` as input
        self._start_embedding = nn.Parameter(torch.Tensor(1, self.decoder_input_dim))
        nn.init.normal_(self._start_embedding, std=0.01)

        # The end embedding which we prepend to the memmory so that the pointernet
        # can learn to terminate the prediction if necessary
        self._end_embedding = nn.Parameter(torch.Tensor(1, self.encoder_output_dim), requires_grad=True)
        nn.init.normal_(self._end_embedding, std=0.01)

        # We set the eos index to zero since the end embedding is prepended to the memory
        self._eos_idx = 0
        self._predict_eos = predict_eos

        def get_helper(helper_type):
            if helper_type == 'teacher_forcing':
                return TeacherForcingHelper(eos_idx=self._eos_idx,
                                            max_steps=max_decoding_steps)
            elif helper_type == 'sample':
                return SamplingHelper(eos_idx=self._eos_idx,
                                      max_steps=max_decoding_steps)
            elif helper_type == 'greedy':
                return GreedyHelper(eos_idx=self._eos_idx,
                                    max_steps=max_decoding_steps)
            elif helper_type == 'beamsearch':
                return BeamSearchHelper(beam_size=beam_size,
                                        eos_idx=self._eos_idx,
                                        max_steps=max_decoding_steps)
            else:
                raise ValueError("Unsupported Decoding Helper!")
        # Training decoding helper
        self._train_helper_type = train_helper
        self._train_decoding_helper = get_helper(train_helper)
        # At prediction time, we'll use a beam search to find the best target sequence.
        self._eval_helper_type = val_helper
        self._eval_decoding_helper = get_helper(val_helper)

    @overrides
    def forward(self,  # type: ignore
                memory: torch.Tensor,
                memory_mask: torch.Tensor,
                init_hidden_state: torch.Tensor = None,
                aux_input: torch.Tensor = None,
                transition_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        memory : ``torch.FloatTensor``, required
            The output of an encoder. Shape: ``[batch_size, memory_len, mem_span_length, encoder_output_dim]``
        memory_mask : ``torch.FloatTensor``, required
            The mask of the encoder output. Shape: ``[batch_size, memory_len, mem_span_length]``
        init_hidden_state : ``torch.FloatTensor``, optional
            The initial hidden state for the decoder cell. Shape: ``[batch_size, decoder_output_dim]``.
            If not provided, the hidden state will be initialized with zero.
        aux_input : ``torch.FloatTensor``, optional
            The auxilary information that will be feed to the deocder in every step.
            Shape: ``[batch_size, aux_input_dim]``.
        transition_mask : ``torch.FloatTensor``, optional
            The mask for restricting the action space. Shape: ``[batch_size, memory_len, memory_len]``
        labels : ``torch.IntTensor``, optional
            The true labels for teacher forcing. Shape: ``[batch_size, decoding_len]``
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch_size = memory.size(0)
        start_state = self._init_decoder_state(memory, memory_mask, init_hidden_state, aux_input, transition_mask)
        # shape: (batch_size, decoder_input_dim)
        start_embedding = self._start_embedding.expand(batch_size, self.decoder_input_dim)
        if self.training:
            helper = self._train_decoding_helper
        else:
            helper = self._eval_decoding_helper
        # shape (all_predictions): (batch_size, K, num_decoding_steps)
        # shape (all_logprobs): (batch_size, K, num_decoding_steps)
        # shape (seq_logprobs): (batch_size, K)
        # shape (final_hidden): (batch_size, K, decoder_output_dim)
        all_predictions, all_logprobs, seq_logprobs, final_hidden = helper.search(
                start_embedding, start_state, self._decoder_step, labels)
        # add a extra beam dimension if needed
        if all_predictions.dim() == 2:
            all_predictions = all_predictions.unsqueeze(1)
            all_logprobs = all_logprobs.unsqueeze(1)
            seq_logprobs = seq_logprobs.unsqueeze(1)
            final_hidden = final_hidden.unsqueeze(1)
        return all_predictions, all_logprobs, seq_logprobs, final_hidden

    def _init_decoder_state(self,
                            tok_memory: torch.Tensor,
                            tok_memory_mask: torch.Tensor,
                            init_hidden_state: torch.Tensor = None,
                            aux_input: torch.Tensor = None,
                            transition_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Initialize the state to be passed to the decoder at the first decoding time step.
        """
        batch_size, memory_len, mem_span_len = tok_memory_mask.size()
        # Prepend the end embedding to the memory so that the model
        # can terminate the decoding by attend on the end vector
        # shape: (batch_size, 1, 1, encoder_output_dim) -> (batch_size, 1, mem_span_len, encoder_output_dim)
        # shape: (batch_size, 1+memory_len, mem_span_len, encoder_output_dim)
        prepend = self._end_embedding.expand(batch_size, self.encoder_output_dim)[:, None, None, :]
        prepend = F.pad(prepend, (0, 0, 0, mem_span_len - 1), 'constant', 0)
        tok_memory = torch.cat([prepend, tok_memory], dim=1)
        # The tok memory mask also need to be prepended
        # shape: (batch_size, 1, mem_span_len)
        # shape: (batch_size, 1+memory_len, mem_span_len)
        tok_mask_prepend = tok_memory_mask.new_zeros((batch_size, 1, mem_span_len))
        tok_mask_prepend[:, :, 0] = 1
        tok_memory_mask = torch.cat([tok_mask_prepend, tok_memory_mask], dim=1)
        # The memory mask also need to be prepended,
        # but here the mask is produced by tok_memory_mask, which alreadly has been prepended.
        # As a result, we only consider changing the mask of eos according to ``predict_eos``
        # shape: (batch_size, 1+memory_len)
        memory_mask = (torch.sum(tok_memory_mask, dim=-1) >= 1).float()
        if not self._predict_eos:
            memory_maks[:, 0] = 0

        # Initialize the decoder hidden state with zeros if not provided.
        # shape: (batch_size, decoder_output_dim)
        if init_hidden_state is None:
            init_hidden_state = memory.new_zeros(batch_size, self.decoder_output_dim)

        state = {"tok_memory":      tok_memory,
                 "tok_memory_mask": tok_memory_mask,
                 "memory_mask": memory_mask,
                 "cell_hidden": init_hidden_state}

        if self._cell == 'lstm':
            # Initialize the decoder context with zeros. Only do it when the cell type is ``LSTMCell``
            # shape: (batch_size, decoder_output_dim)
            init_context = tok_memory.new_zeros(batch_size, self.decoder_output_dim)
            state["cell_context"] = init_context

        if not aux_input is None:
            state['aux_input'] = aux_input

        # If transition mask is provided, we also need to prepend ones on the row and column axis
        # to account for the eos.
        if not transition_mask is None:
            # shape: (batch_size, 1+memory_len, 1+memory_len)
            transition_mask = F.pad(transition_mask, (1, 0, 1, 0), 'constant', int(self._predict_eos))
            state['transition_mask'] = transition_mask

        return state

    def _decoder_step(self,
                      state: Dict[str, torch.Tensor],
                      last_predictions: torch.Tensor = None,
                      next_input: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # shape: (group_size, memory_len, mem_span_len, encoder_output_dim)
        tok_memory = state["tok_memory"]
        # shape: (group_size, memory_len, mem_span_len)
        tok_memory_mask = state["tok_memory_mask"].float()
        # shape: (group_size, memory_len)
        memory_mask = state["memory_mask"].float()
        group_size, memory_len, mem_span_len, _ = tok_memory.size()

        if not last_predictions is None:
            # key ``memory`` should have been set at timestep > 0
            # shape: (group_size, memory_len, encoder_output_dim)
            memory = state["memory"]
            # We obtain the next input by last predictions
            # The last predictions represent the indices pointing to encoder outputs.
            # We gather the encoder outputs w.r.t the last predictions and take them as the next inputs
            # shape: (group_size, 1, encoder_output_dim)
            expand_last_predictions = last_predictions[:, None, None].expand(group_size, 1, self.encoder_output_dim)
            # shape: (group_size, encoder_output_dim)
            decoder_input = torch.gather(memory, 1, expand_last_predictions).squeeze(1)

            # We also gather the transition restriction, if provided, w.r.t the last predictions,
            # and apply on the memory mask.
            # shape: (group_size, memory_len)
            if not state.get("transition_mask", None) is None:
                # shape: (group_size, 1, memory_len)
                expand_last_predictions = last_predictions[:, None, None].expand(group_size, 1, memory_len)
                transition_mask = torch.gather(state["transition_mask"], 1, expand_last_predictions).squeeze(1)
                memory_mask = memory_mask * transition_mask.float()
        else:
            # We use ``next input`` as the next decoder input. This condition is use for the initial step.
            assert not next_input is None
            # shape: (group_size, decoder_input_dim)
            decoder_input = next_input

        # We apply another mask to prevent the model repeatedly selecting the same memory to point to.
        if "hist_predictions" in state:
            hist_attention = tok_memory.new_zeros((group_size, memory_len))
            hist_attention.scatter_(1, state["hist_predictions"], 1.)
            memory_mask = memory_mask * (1. - hist_attention)

        # shape: (group_size, encoder_output_dim + aux_input_dim)
        if not state.get('aux_input', None) is None:
            decoder_input = torch.cat((decoder_input, state['aux_input']), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        if self._cell == 'lstm':
            # shape: (group_size, decoder_output_dim), (group_size, decoder_output_dim)
            next_cell_hidden, next_cell_context = self._decoder_cell(
                    projected_decoder_input,
                    (state["cell_hidden"], state["cell_context"]))
        elif self._cell == 'gru':
            # shape: (group_size, decoder_output_dim)
            next_cell_hidden = self._decoder_cell(
                    projected_decoder_input,
                    state["cell_hidden"])
        if not last_predictions is None:
            # Here we are finding any sequences where we predicted the end token in
            # the previous timestep and directly passing through its hidden state.
            # shape: (group_size, decoder_output_dim)
            expand_last_predictions = last_predictions.unsqueeze(-1).expand(
                    group_size,
                    self.decoder_output_dim
            )
            # shape: (group_size, decoder_output_dim)
            state["cell_hidden"] = torch.where(
                    expand_last_predictions == self._eos_idx,
                    state["cell_hidden"],
                    next_cell_hidden
            )
            if self._cell == 'lstm':
                # shape: (group_size, decoder_output_dim)
                state["cell_context"] = torch.where(
                        expand_last_predictions == self._eos_idx,
                        state["cell_context"],
                        next_cell_context
                )

        # Compute the alignments scores w.r.t the each token embeddings
        # shape: (group_size, 1, decoder_output_dim)
        query = state["cell_hidden"].unsqueeze(1)
        flatten_tok_memory = tok_memory.reshape(group_size, memory_len*mem_span_len, -1)
        # shape: (group_size, memory_len, mem_span_len)
        tok_attention_matrix = self._fg_attention(query, flatten_tok_memory).\
                squeeze(1).\
                reshape(group_size, memory_len, mem_span_len)
        tok_attention_probs = util.masked_softmax(tok_attention_matrix, tok_memory_mask)
        # shape: (group_size, memory_len)
        state['memory'] = util.weighted_sum(tok_memory, tok_attention_probs)
        # shape: (group_size, memory_len)
        attention_matrix = self._attention(query, state['memory']).squeeze(1)
        attention_logprobs = util.masked_log_softmax(attention_matrix, memory_mask)
        #attention_probs = util.masked_softmax(attention_matrix, memory_mask)
        return state, attention_logprobs#, attention_probs
