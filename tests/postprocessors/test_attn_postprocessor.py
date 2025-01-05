from unittest import TestCase

import torch
from lightning_ocr.dictionary.dictionary import Dictionary
from lightning_ocr.models.postprocessors.attn_postprocessor import (
    AttentionPostprocessor,
)


class TestAttentionPostprocessor(TestCase):
    def test_call(self):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        # test diction cfg
        dict_gen = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=False,
        )
        data_samples = [{"gt_text": "122"}]
        postprocessor = AttentionPostprocessor(
            max_seq_len=None, dictionary=dict_gen, ignore_chars=["0"]
        )
        dict_gen.end_idx = 3
        # test decode output to index
        dummy_output = torch.Tensor(
            [
                [
                    [1, 100, 3, 4, 5, 6, 7, 8],
                    [100, 2, 3, 4, 5, 6, 7, 8],
                    [1, 2, 100, 4, 5, 6, 7, 8],
                    [1, 2, 100, 4, 5, 6, 7, 8],
                    [100, 2, 3, 4, 5, 6, 7, 8],
                    [1, 2, 3, 100, 5, 6, 7, 8],
                    [100, 2, 3, 4, 5, 6, 7, 8],
                    [1, 2, 3, 100, 5, 6, 7, 8],
                ]
            ]
        )
        data_samples = postprocessor(dummy_output, data_samples)
        self.assertEqual(data_samples[0]["pred_text"], "122")
