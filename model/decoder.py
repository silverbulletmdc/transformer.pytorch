import torch
from torch import nn
from .attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, dim_model):
        super(Decoder, self).__init__()
        self.mha_self = MultiHeadAttention(dim_model, dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.mha_cross = MultiHeadAttention(dim_model, dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_model)
        )
        self.layer_norm3 = nn.LayerNorm(dim_model)


    def forward(self, input:torch.Tensor, output:torch.Tensor):
        """
        Transformer Decoder
        :param input: [B, L, d_model] The output from encoder
        :param output: [B, L, d_model] The shifted outputs
        :return:
        """
        B, L = output.shape[:2]
        mask = torch.ones(L, L, dtype=torch.uint8).tril()
        output = self.mha_self(output, output, output, mask) + output
        output = self.layer_norm1(output)
        output = self.mha_cross(input, input, output) + output
        output = self.layer_norm2(output)
        output = self.ff(output) + output
        output = self.layer_norm3(output)
        return output



