import torch
import unittest
from lightning_ocr.models.encoders.abi_encoder import ABIEncoder

class TestABIEncoder(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            ABIEncoder(d_model=512, n_head=10)

    def test_forward(self):
        model = ABIEncoder()
        x = torch.randn(10, 512, 8, 32)
        self.assertEqual(model(x).shape, torch.Size([10, 512, 8, 32]))