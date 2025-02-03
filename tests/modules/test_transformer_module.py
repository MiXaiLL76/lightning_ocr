from unittest import TestCase
import torch
from lightning_ocr.modules.layers.transformer_module import PositionalEncoding


class TestPositionalEncoding(TestCase):
    def test_forward(self):
        pos_encoder = PositionalEncoding()
        x = torch.rand(1, 30, 512)
        out = pos_encoder(x)
        assert out.size() == x.size()
