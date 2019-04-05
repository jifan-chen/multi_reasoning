from typing import Dict, Tuple, List, Any, Union, Callable
import warnings

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch import nn

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
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size,)``.
        """
        raise NotImplementedError


class GreedyHelper(DecodeHelper_):
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.max(logits, dim=1)

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_embedding.size()[0]

        # List of (batch_size,) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

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
            return start_predicted_classes.unsqueeze(-1), start_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size,)
        last_log_probabilities = start_log_probabilities

        # shape: [(batch_size,)]
        predictions.append(start_predicted_classes)
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

        return all_predictions, last_log_probabilities


class SamplingHelper(GreedyHelper):
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, 1)
        samples = torch.multinomial(torch.exp(logits), 1)
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

    def search(self,
               start_embedding: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
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
        start_top_log_probabilities, start_predicted_classes = self._sample(start_class_log_probabilities)
        if self.beam_size == 1 and (start_predicted_classes == self._eos_idx).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

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

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        assert (all_predictions == state['hist_predictions'].view(batch_size, self.beam_size, -1)).all()

        return all_predictions, last_log_probabilities


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
                 beam_size: int = 3,
                 max_decoding_steps: int = 5) -> None:
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
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # At initial step we take the trainable ``start embedding`` as input
        self._start_embedding = nn.Parameter(torch.Tensor(1, self.decoder_input_dim))
        nn.init.normal_(self._start_embedding, std=0.01)

        # The end embedding which we prepend to the memmory so that the pointernet
        # can learn to terminate the prediction if necessary
        self._end_embedding = nn.Parameter(torch.Tensor(1, self.encoder_output_dim), requires_grad=True)
        nn.init.normal_(self._end_embedding, std=0.01)

        # We set the eos index to zero since the end embedding is prepended to the memory
        self._eos_idx = 0

        # Training decoding helper
        self._train_decoding_helper = SamplingHelper(eos_idx=self._eos_idx,
                                                     max_steps=max_decoding_steps)
        # At prediction time, we'll use a beam search to find the best target sequence.
        self._eval_decoding_helper = BeamSearchHelper(beam_size=beam_size,
                                                      eos_idx=self._eos_idx,
                                                      max_steps=max_decoding_steps)

    @overrides
    def forward(self,  # type: ignore
                memory: torch.Tensor,
                memory_mask: torch.Tensor,
                init_hidden_state: torch.Tensor = None,
                aux_input: torch.Tensor = None,
                transition_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Returns
        -------
        Dict[str, torch.Tensor]
        """
        batch_size = memory.size(0)
        start_state = self._init_decoder_state(memory, memory_mask, init_hidden_state, aux_input, transition_mask)
        # shape: (batch_size, decoder_input_dim)
        start_embedding = self._start_embedding.expand(batch_size, self.decoder_input_dim)
        if self.training:
            # shape (all_predictions): (batch_size, num_decoding_steps)
            # shape (seq_logprobs): (batch_size,)
            all_predictions, seq_logprobs = self._train_decoding_helper.search(
                    start_embedding, start_state, self._decoder_step)
        else:
            # shape (all_predictions): (batch_size, beam_size, num_decoding_steps)
            # shape (seq_logprobs): (batch_size, beam_size)
            all_predictions, seq_logprobs = self._eval_decoding_helper.search(
                    start_embedding, start_state, self._decoder_step)
        return all_predictions, seq_logprobs

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
        # The memory mask also need to be prepended with ones
        # shape: (batch_size, 1)
        # shape: (batch_size, 1+memory_len)
        mask_prepend = memory_mask.new_ones((batch_size, 1))
        memory_mask = torch.cat([mask_prepend, memory_mask], dim=1)

        # Initialize the decoder hidden state with zeros if not provided.
        # shape: (batch_size, decoder_output_dim)
        if init_hidden_state is None:
            init_hidden_state = memory.new_zeros(batch_size, self.decoder_output_dim)

        # Initialize the decoder context with zeros.
        # shape: (batch_size, decoder_output_dim)
        init_context = memory.new_zeros(batch_size, self.decoder_output_dim)

        state = {"memory":      memory,
                 "memory_mask": memory_mask,
                 "cell_hidden": init_hidden_state,
                 "cell_context":init_context}
        if not aux_input is None:
            state['aux_input'] = aux_input
        # If transition mask is provided, we also need to prepend ones on the row and column axis
        # to account for the eos.
        if not transition_mask is None:
            # shape: (batch_size, memory_len, 1+memory_len)
            mask_prepend = transition_mask.new_ones((batch_size, memory_len, 1))
            transition_mask = torch.cat([mask_prepend, transition_mask], dim=2)
            # shape: (batch_size, 1+memory_len, 1+memory_len)
            mask_prepend = transition_mask.new_ones((batch_size, 1, 1+memory_len))
            transition_mask = torch.cat([mask_prepend, transition_mask], dim=1)
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

        # shape: (group_size, decoder_output_dim), (group_size, decoder_output_dim)
        state["cell_hidden"], state["cell_context"] = self._decoder_cell(
                projected_decoder_input,
                (state["cell_hidden"], state["cell_context"]))

        # Compute the alignments scores w.r.t the encoder outputs
        # shape: (group_size, 1, decoder_output_dim)
        query = state["cell_hidden"].unsqueeze(1)
        # shape: (group_size, memory_len)
        attention_matrix = self._attention(query, memory).squeeze(1)
        attention_logprobs = util.masked_log_softmax(attention_matrix, memory_mask)
        #attention_probs = util.masked_softmax(attention_matrix, memory_mask)
        return state, attention_logprobs#, attention_probs
