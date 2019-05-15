from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from allennlp.modules import Seq2SeqEncoder


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


@Seq2SeqEncoder.register("bi_attention")
class BiAttention(Seq2SeqEncoder):
    def __init__(self, input_size, dropout, output_projection_dim=None):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self._output_dim = output_projection_dim or input_size
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.output_linear = nn.Sequential(
                nn.Linear(input_size * 4, self._output_dim),
                nn.ReLU()
            )

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory=None, mask=None, mask_sp=None, att_sup_metric=None):
        if memory is None:
            memory = input
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        if not mask_sp is None:
            # print(att.shape, mask_sp.shape)
            loss = torch.mean(-torch.log(torch.sum(weight_one * mask_sp.unsqueeze(1), dim=-1) + 1e-30))
        else:
            loss = None
        outputs = torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
        outputs = self.output_linear(outputs)
        return outputs, loss, weight_one
