import torch
from overrides import overrides

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util
from allennlp.nn.util import get_range_vector, get_device_of
from allennlp.modules import Seq2SeqEncoder


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets
    # print(offset_indices)
    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


# @SpanExtractor.register("self_attentive")
class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.

    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self,
                 input_dim: int,
                 span_self_attentive_encoder: Seq2SeqEncoder,
                 dep_masks: torch.IntTensor = None) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))
        self._span_self_attentive_encoder = span_self_attentive_encoder

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        # print(sequence_tensor.shape)
        # print(span_indices)
        # for x in span_indices:
        #     print(x)
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)
        has_span_mask = (span_starts > -1).float()

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_mask = (max_span_range_indices <= span_widths).float()
        span_mask = span_mask * has_span_mask
        # print(torch.sum(span_mask, dim=[-1, -2]))
        # raw_span_indices = span_ends - max_span_range_indices
        raw_span_indices = span_starts + max_span_range_indices
        # print('raw_span_indices:', raw_span_indices)
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        # span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
        span_indices = span_indices * span_mask.long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size * num_spans, max_batch_span_width, embedding_dim)
        batch_size, num_spans, max_batch_span_width = span_indices.size()
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)\
            .view(batch_size * num_spans, max_batch_span_width, -1)
        span_dep_masks = util.batched_index_select(span_indices_mask, span_indices, flat_span_indices)\
            .view(batch_size * num_spans, max_batch_span_width, -1)
        print('span_dep_masks:', span_dep_masks)
        attended_span_embeddings = self._span_self_attentive_encoder(span_embeddings,
                                                                     span_mask.view(batch_size * num_spans, -1))

        # print(attended_span_embeddings.shape)
        # We split the context into sentences and now we want to reconstruct the context using the sentences
        # Shape: (batch_size, num_sentences, max_batch_sent_length, embedding_dim) ->
        # (batch_size, max_batch_context_length, embedding_dim )
        recovered_indices = []
        for slice_embs, m in zip(attended_span_embeddings.view(batch_size, num_spans, max_batch_span_width, -1), span_mask.byte()):
            # print('selected_shape:', torch.masked_select(slice_embs, m.unsqueeze(-1)).shape)
            # print(torch.masked_select(slice_embs, m.unsqueeze(-1)).view(-1, 200).shape)
            recovered_indices.append(torch.masked_select(slice_embs, m.unsqueeze(-1)))
        recovered_context_representation = torch.nn.utils.rnn.pad_sequence(recovered_indices, batch_first=True)
        # print(recovered_context_representation.shape)
        recovered_context_representation = recovered_context_representation.view(batch_size, sequence_tensor.size(1), -1)
        # print(recovered_context_representation.shape)

        return recovered_context_representation
