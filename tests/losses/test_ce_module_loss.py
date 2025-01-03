from unittest import TestCase

import torch
from lightning_ocr.models.module_losses.ce_module_loss import CEModuleLoss
from lightning_ocr.dictionary.dictionary import Dictionary


class TestCEModuleLoss(TestCase):
    def setUp(self) -> None:
        data_sample1 = {
            "gt_text": "hello",
        }
        data_sample2 = {
            "gt_text": "01abyz",
        }
        data_sample3 = {
            "gt_text": "123456789",
        }
        self.gt = [data_sample1, data_sample2, data_sample3]

    def test_init(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False,
        )

        with self.assertRaises(AssertionError):
            CEModuleLoss(dict_cfg, reduction=1)
        with self.assertRaises(AssertionError):
            CEModuleLoss(dict_cfg, reduction="avg")
        with self.assertRaises(AssertionError):
            CEModuleLoss(dict_cfg, flatten=1)
        with self.assertRaises(AssertionError):
            CEModuleLoss(dict_cfg, ignore_first_char=1)
        with self.assertRaises(AssertionError):
            CEModuleLoss(dict_cfg, ignore_char=["ignore"])
        ce_loss = CEModuleLoss(dict_cfg)
        self.assertEqual(ce_loss.ignore_index, 37)
        ce_loss = CEModuleLoss(dict_cfg, ignore_char=-1)
        self.assertEqual(ce_loss.ignore_index, -1)
        # with self.assertRaises(ValueError):
        with self.assertWarns(UserWarning):
            ce_loss = CEModuleLoss(dict_cfg, ignore_char="ignore")
        with self.assertWarns(UserWarning):
            ce_loss = CEModuleLoss(
                Dictionary(chars, with_unknown=True), ignore_char="M", pad_with="none"
            )
        with self.assertWarns(UserWarning):
            ce_loss = CEModuleLoss(
                Dictionary(chars, with_unknown=False), ignore_char="M", pad_with="none"
            )
        with self.assertWarns(UserWarning):
            ce_loss = CEModuleLoss(
                Dictionary(chars, with_unknown=False),
                ignore_char="unknown",
                pad_with="none",
            )
        ce_loss = CEModuleLoss(dict_cfg, ignore_char="1")
        self.assertEqual(ce_loss.ignore_index, 1)

    def test_forward(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False,
        )

        max_seq_len = 40
        ce_loss = CEModuleLoss(dict_cfg)
        ce_loss.get_targets(self.gt)
        outputs = torch.rand(3, max_seq_len, ce_loss.dictionary.num_classes)
        losses = ce_loss(outputs, self.gt)
        self.assertIsInstance(losses, dict)
        self.assertIn("loss_ce", losses)
        self.assertEqual(losses["loss_ce"].size(1), max_seq_len)

        # test ignore_first_char
        ce_loss = CEModuleLoss(dict_cfg, ignore_first_char=True)
        ignore_first_char_losses = ce_loss(outputs, self.gt)
        self.assertEqual(
            ignore_first_char_losses["loss_ce"].shape, torch.Size([3, max_seq_len - 1])
        )

        # test flatten
        ce_loss = CEModuleLoss(dict_cfg, flatten=True)
        flatten_losses = ce_loss(outputs, self.gt)
        self.assertEqual(flatten_losses["loss_ce"].shape, torch.Size([3 * max_seq_len]))

        self.assertTrue(
            torch.isclose(
                losses["loss_ce"].view(-1), flatten_losses["loss_ce"], atol=1e-6, rtol=0
            ).all()
        )
