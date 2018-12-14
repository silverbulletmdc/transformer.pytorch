import sys
sys.path.append('..')
from model.attention import ScaledDotProductAttention, MultiHeadAttention
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer import Transformer
import unittest
import torch


def assert_list_equal(l1, l2):
    assert len(l1) == len(l2)
    for i1, i2 in zip(l1, l2):
        assert i1 == i2


class MyTestCase(unittest.TestCase):

    def test_ScaledDotProductAttention(self):
        sdpa = ScaledDotProductAttention()
        q = torch.randn(5, 4, 512)
        v = torch.randn(5, 10, 1024)
        k = torch.randn(5, 10, 512)
        assert_list_equal(sdpa(q, k, v)[0].shape, [5, 4, 1024])

    def test_MultiHeadAttention(self):
        mha = MultiHeadAttention(512, 1024)
        q = torch.randn(5, 4, 512)
        v = torch.randn(5, 10, 1024)
        k = torch.randn(5, 10, 512)
        assert_list_equal(mha(v, k, q).shape, [5, 4, 1024])

    def test_Encoder(self):
        encoder = Encoder(512)
        v = torch.randn(5, 10, 512)
        assert_list_equal(encoder(v).shape, v.shape)

    def test_Decoder(self):
        decoder = Decoder(512)
        encoder_output = torch.randn(5, 10, 512)
        output = torch.randn(5, 8, 512)
        assert_list_equal(decoder(encoder_output, output).shape, [5, 8, 512])

    def test_PositionalEncoder(self):
        transfomer = Transformer(512, 10)
        pe = transfomer.get_positional_encoder(10)
        assert_list_equal(pe.shape, [10, 512])

    def test_Transformer(self):
        transfomer = Transformer(512, 10)
        input = torch.randn(5, 10, 512)
        output = torch.randn(5, 8, 512)
        output = transfomer(input, output)
        assert_list_equal(output.shape, [5, 8, 10])


if __name__ == '__main__':
    unittest.main()
