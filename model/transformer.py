import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, d_input, classes, dim_model = 512, d_v = 1024, N = 6):
        super(Transformer, self).__init__()
        self.d_input = d_input
        self.dim_model = dim_model
        self.d_v = d_v
        self.classes = classes
        # Input embedding
        self.input_embedding = nn.Embedding(d_input, dim_model)
        self.output_embedding = nn.Embedding(classes, dim_model)

        # Build encoder and decoder
        N = 6

        self.encoders = nn.Sequential(*[Encoder(dim_model) for i in range(N)])
        self.decoders = nn.Sequential(*[Decoder(dim_model) for i in range(N)])

        self.output_linear = nn.Linear(dim_model, classes)
        self.softmax = nn.Softmax(dim=2)

    def get_positional_encoder(self, L):
        positional_encoder = torch.empty([L, self.dim_model], dtype=torch.float32)
        pos = torch.arange(L, dtype=torch.float32)
        i = torch.arange(self.dim_model / 2, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(pos, i)
        pe_even = torch.sin(
            grid_x / 10000**((2 * grid_y) / self.dim_model)
        )
        pe_odd = torch.cos(
            grid_x / 10000**((2 * grid_y) / self.dim_model)
        )
        positional_encoder[:, ::2] = pe_even
        positional_encoder[:, 1::2] = pe_odd

        return positional_encoder

    def forward(self, input:torch.Tensor, output:torch.Tensor)->torch.Tensor:
        """

        Transformer

        :param input: [B, L_in, d_model]
        :param output: [B, L_out, d_model]
        :return:
            [B, L_out, classes] output of transformer.
        """
        input = self.input_embedding(input)

        # construct positional encoder
        positional_encoder_input = self.get_positional_encoder(input.shape[1])

        input += positional_encoder_input
        input = self.encoders(input)

        output = self.output_embedding(output)
        positional_encoder_output = self.get_positional_encoder(output.shape[1])

        output += positional_encoder_output
        for decoder in self.decoders:
            output = decoder(input, output)

        output = self.output_linear(output)
        output = self.softmax(output)

        return output
