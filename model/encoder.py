import torch
from torch import nn
from .attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, dim_model):
        super(Encoder, self).__init__()
        self.dim_model = dim_model
        self.mha = MultiHeadAttention(dim_model, dim_model)
        self.layer_norm_mha = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_model)
        )
        self.layer_norm_ff = nn.LayerNorm(dim_model)

    def forward(self, input: torch.Tensor)->torch.Tensor:
        """
        Transformer encoder

        :param input: [B, L, d_model]
        :return: [B, L, d_model]
        """
        attention = self.mha(input, input, input)
        input += attention
        input = self.layer_norm_mha(input)
        output = self.ff(input)
        output += input
        output = self.layer_norm_ff(output)
        return output



