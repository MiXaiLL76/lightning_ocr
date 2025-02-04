from unittest import TestCase
import torch
from lightning_ocr.modules.decoders.position_attention_decoder import (
    PositionAttentionDecoder,
)


class TestPositionAttentionDecoder(TestCase):
    def setUp(self):
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.num_classes = len(self.chars) + 3  # BOS , EOS , PAD

    def test_init(self):
        decoder = PositionAttentionDecoder(
            num_classes=self.num_classes, return_feature=False
        )
        self.assertIsInstance(decoder.prediction, torch.nn.Linear)

    def test_forward(self):
        feat = torch.randn(2, 512, 8, 8)
        encoder_out = torch.randn(2, 128, 8, 8)
        decoder = PositionAttentionDecoder(
            num_classes=self.num_classes, return_feature=False
        )
        output = decoder(feat=feat, out_enc=encoder_out)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 39))

        decoder = PositionAttentionDecoder(num_classes=self.num_classes)
        output = decoder(feat=feat, out_enc=encoder_out)
        self.assertTupleEqual(tuple(output.shape), (2, 40, 512))

        feat_new = torch.randn(2, 256, 8, 8)
        with self.assertRaises(AssertionError):
            decoder(feat_new, encoder_out)

        encoder_out_new = torch.randn(2, 256, 8, 8)
        with self.assertRaises(AssertionError):
            decoder(feat, encoder_out_new)
