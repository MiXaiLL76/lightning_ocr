from unittest import TestCase
import numpy as np
import torch
from lightning_ocr.models.module_losses.abi_module_loss import ABIModuleLoss
from lightning_ocr.dictionary.dictionary import Dictionary


class TestABIModuleLoss(TestCase):
    def setUp(self) -> None:
        data_sample1 = {"gt_text" : "hello"}
        data_sample2 = {"gt_text" : "123"}
        self.gt = [data_sample1, data_sample2]

    def _equal(self, a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    def test_forward(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False)
        abi_loss = ABIModuleLoss(dict_cfg, max_seq_len=10)
        abi_loss.get_targets(self.gt)
        outputs = dict(
            out_vis=dict(logits=torch.randn(2, 10, 38)),
            out_langs=[
                dict(logits=torch.randn(2, 10, 38)),
                dict(logits=torch.randn(2, 10, 38))
            ],
            out_fusers=[
                dict(logits=torch.randn(2, 10, 38)),
                dict(logits=torch.randn(2, 10, 38))
            ])
        losses = abi_loss(outputs, self.gt)
        self.assertIsInstance(losses, dict)
        self.assertIn('loss_visual', losses)
        self.assertIn('loss_lang', losses)
        self.assertIn('loss_fusion', losses)

        outputs.pop('out_vis')
        abi_loss(outputs, self.gt)
        out_langs = outputs.pop('out_langs')
        abi_loss(outputs, self.gt)
        outputs.pop('out_fusers')
        with self.assertRaises(AssertionError):
            abi_loss(outputs, self.gt)
        outputs['out_langs'] = out_langs
        abi_loss(outputs, self.gt)