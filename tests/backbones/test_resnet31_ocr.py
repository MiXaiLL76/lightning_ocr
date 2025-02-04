from unittest import TestCase

import torch

from lightning_ocr.modules.backbones.resnet31_ocr import ResNet31OCR


class TestResNet31OCR(TestCase):
    def test_forward(self):
        """Test resnet backbone."""
        with self.assertRaises(ValueError):
            ResNet31OCR(2.5)

        with self.assertRaises(TypeError):
            ResNet31OCR(3, layers=5)

        with self.assertRaises(TypeError):
            ResNet31OCR(3, channels=5)

        # Test ResNet18 forward
        model = ResNet31OCR()
        model.train()

        imgs = torch.randn(1, 3, 32, 160)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 512, 4, 40]))
