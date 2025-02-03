import unittest

import torch

from lightning_ocr.modules.encoders.channel_reduction_encoder import ChannelReductionEncoder


class TestChannelReductionEncoder(unittest.TestCase):

    def setUp(self):
        self.feat = torch.randn(2, 512, 8, 25)

    def test_encoder(self):
        encoder = ChannelReductionEncoder(512, 256)
        encoder.train()
        out_enc = encoder(self.feat)
        self.assertEqual(out_enc.shape, torch.Size([2, 256, 8, 25]))