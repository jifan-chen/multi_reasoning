import torch
from allennlp.nn import util
from allennlp.nn.util import get_range_vector, get_device_of


def convert_sequence_to_spans(sequence_tensor: torch.FloatTensor,
                              span_indices: torch.LongTensor):
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = span_indices.split(1, dim=-1)
    # Shape: (batch_size, num_spans, 1)
    has_span_mask = (span_starts > -1).float()

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
    raw_span_indices = span_starts + max_span_range_indices

    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the sequence
    # have an end index < max_batch_span_width, so we add this to the mask here.
    # span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    span_indices = span_indices * span_mask.long()

    # Shape: (batch_size * num_spans * max_batch_span_width)
    flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

    # Shape: (batch_size * num_spans, max_batch_span_width, embedding_dim)
    batch_size, num_spans, max_batch_span_width = span_indices.size()
    span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices) \
        .view(batch_size * num_spans, max_batch_span_width, -1)

    return span_embeddings, span_mask


def convert_span_to_sequence(sequence_tensor: torch.FloatTensor,
                             spans_tensor: torch.FloatTensor,
                             span_mask: torch.LongTensor):
    batch_size, num_spans, max_batch_span_width = span_mask.size()
    recovered_indices = []
    for slice_embs, m in zip(spans_tensor.view(batch_size, num_spans, max_batch_span_width, -1),
                             span_mask.byte()):
        # print('selected_shape:', torch.masked_select(slice_embs, m.unsqueeze(-1)).shape)
        # print(torch.masked_select(slice_embs, m.unsqueeze(-1)).view(-1, 200).shape)
        recovered_indices.append(torch.masked_select(slice_embs, m.unsqueeze(-1)))
    recovered_context_representation = torch.nn.utils.rnn.pad_sequence(recovered_indices, batch_first=True)
    recovered_context_representation = recovered_context_representation.view(batch_size, sequence_tensor.size(1), -1)
    return recovered_context_representation

