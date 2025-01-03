from unittest import TestCase

import torch

from lightning_ocr.models.decoders.abi_vision_decoder import ABIVisionDecoder


class TestABIVisionDecoder(TestCase):
    def test_forward(self):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        decoder = ABIVisionDecoder(len(chars), in_channels=32, num_channels=16, max_seq_len=10)

        out_enc = torch.randn(2, 32, 8, 32)
        result = decoder(out_enc)
        self.assertIn('feature', result)
        self.assertIn('logits', result)
        self.assertIn('attn_scores', result)
        self.assertEqual(result['feature'].shape, torch.Size([2, 10, 32]))
        self.assertEqual(result['logits'].shape, torch.Size([2, 10, len(chars)]))
        self.assertEqual(result['attn_scores'].shape,
                            torch.Size([2, 10, 8, 32]))
