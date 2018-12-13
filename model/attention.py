import torch
from torch import nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.5):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """
        Scaled Dot-Product Attention

        :param q: [B, L_q, d_k] query
        :param k: [B, L, d_k] key
        :param v: [B, L, d_v] value
        :return:
            [B, L_q, d_v] attention
        """
        d_k = q.shape[2]
        scaled = torch.bmm(q, k.transpose(1, 2))/ np.sqrt(d_k)
        if mask is not None:
            scaled.masked_fill_(mask, -np.inf)

        attention = self.softmax(scaled)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_k, dim_v, h=8):
        super(MultiHeadAttention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.h = h

        self.linear_layers_k = [nn.Linear(self.dim_k, self.dim_k) for i in range(self.h)]
        self.linear_layers_v = [nn.Linear(self.dim_v, self.dim_v) for i in range(self.h)]
        self.linear_layers_q = [nn.Linear(self.dim_k, self.dim_k) for i in range(self.h)]
        self.sdpa_layers = [ScaledDotProductAttention() for i in range(self.h)]

        self.output_linear = nn.Linear(self.h * dim_v, dim_v)

    def forward(self, v:torch.Tensor, k:torch.Tensor, q:torch.Tensor, mask:torch.Tensor=None):
        """
        Multi Head Attention

        :param v: [B, L_q, d_k] queries
        :param k: [B, L, d_k] key
        :param q: [B, L, d_v] values
        :param mask: [L_q, L] mask
        :return:
            [B, L_q, d_v] attention.
        """
        sdpas = []
        for i in range(self.h):
            v = self.linear_layers_v[i](v)
            k = self.linear_layers_k[i](k)
            q = self.linear_layers_q[i](q)
            sdpa = self.sdpa_layers[i](q, k, v, mask)[0]
            sdpas.append(sdpa)
        output = torch.cat(sdpas, dim=2)
        output = self.output_linear(output)
        return output
