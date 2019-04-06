from overrides import overrides
import torch
from torch.nn import Dropout, Linear

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from my_library.metrics import AttF1Measure


@Seq2SeqEncoder.register("multi_head_self_attention_with_sup")
class MultiHeadSelfAttentionWithSup(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadSelfAttentionWithSup, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim
        self._num_heads_of_supervision = 1

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = Linear(input_dim, 2 * attention_dim + values_dim)
        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None,
                mask_sp: torch.IntTensor = None,
                att_sup_metric: AttF1Measure = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        mask_sp : ``torch.IntTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps, timesteps)

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """

        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)

        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        combined_projection = self._combined_projection(inputs)
        # split by attention dim - if values_dim > attention_dim, we will get more
        # than 3 elements returned. All of the rest are the values vector, so we
        # just concatenate them back together again below.
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()
        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        # attention = masked_softmax(scaled_similarities, mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps))

        attention = masked_softmax(scaled_similarities,
                                   #mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps))
                                   mask[:, None, :].expand(batch_size, num_heads, timesteps).contiguous().view(batch_size * num_heads, timesteps))
        # attention_sp = attention.clone().split(num_heads_strong_sup * batch_size, dim=0)[1]
        attention = attention.view(batch_size, num_heads, timesteps, timesteps)
        attention = attention.transpose(0, 1).contiguous().view(num_heads * batch_size, timesteps, timesteps)
        attention_for_sup, attention_no_sup = torch.split(attention, batch_size, dim=0)
        # mask_sp = torch.cat([mask_sp.repeat(1, self._num_heads_of_supervision),
        #                     mask.repeat(1, num_heads - self._num_heads_of_supervision)],
        #                     dim=-1).view(batch_size * num_heads, 1, timesteps)
        # print(mask.shape)
        # print(mask_sp.shape)
        # print(attention_for_sup.shape)
        # print('attention_sp:', attention_sp.shape)
        # print(torch.sum(mask_sp, dim=-1)[0][:100])
        if not mask_sp is None:
            #loss = torch.mean(-torch.log(torch.sum(attention_for_sup * mask_sp, dim=-1) + 1e-10))
            square_mask = mask[:, None, :] * mask[:, :, None]
            mask_sp = (mask_sp * square_mask).float()
            mask_loss = (torch.sum(mask_sp, dim=-1) > 0).float()
            num_valid_loss = torch.sum(mask_loss)
            if num_valid_loss.item() > 0:
                loss = torch.sum(-torch.log(torch.sum(attention_for_sup * mask_sp, dim=-1) + 1e-10) * mask_loss) / num_valid_loss
            else:
                loss = torch.tensor(0.)
            '''
            tot = torch.sum(mask_sp.float(), dim=-1, keepdim=True)
            attention_for_sup = mask_sp.float() / (tot + (tot == 0).float())
            loss = torch.tensor(0.)
            '''
            '''
            if self.training:
                pass
            else:
                tot = torch.sum(mask_sp.float(), dim=-1, keepdim=True)
                has_coref = (tot == 1).float()
                norm_mask_sp = mask_sp.float() / (tot + (1 - has_coref))
                #attention_for_sup = attention_for_sup * (1 - has_coref) + norm_mask_sp * has_coref
                attention_for_sup = norm_mask_sp
            '''
            att_sup_metric(attention_for_sup, mask_sp, square_mask)
        else:
            loss = None

        '''
        attention = torch.cat([attention_for_sup, attention_no_sup], dim=0)
        '''
        attention = torch.stack([attention_for_sup, attention_no_sup], dim=1)
        attention = attention.view(batch_size * num_heads, timesteps, timesteps)
        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, timesteps, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs, loss, attention.view(batch_size, num_heads, timesteps, -1)
