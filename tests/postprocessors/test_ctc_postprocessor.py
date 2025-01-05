from unittest import TestCase

import torch

from lightning_ocr.dictionary.dictionary import Dictionary
from lightning_ocr.models.postprocessors.ctc_postprocessor import CTCPostProcessor

class TestCTCPostProcessor(TestCase):

    def test_get_single_prediction(self):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        dict_gen = Dictionary(
            chars,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        data_samples = [{}]
        postprocessor = CTCPostProcessor(max_seq_len=None, dictionary=dict_gen)

        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8]]])
        index, score = postprocessor.get_single_prediction(
            dummy_output[0], data_samples[0])
        self.assertListEqual(index, [1, 0, 2, 0, 3, 0, 3])
        self.assertListEqual(score,
                             [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        postprocessor = CTCPostProcessor(
            max_seq_len=None, dictionary=dict_gen, ignore_chars=['0'])
        index, score = postprocessor.get_single_prediction(
            dummy_output[0], data_samples[0])
        self.assertListEqual(index, [1, 2, 3, 3])
        self.assertListEqual(score, [100.0, 100.0, 100.0, 100.0])

    def test_call(self):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        dict_gen = Dictionary(
            chars,
            with_start=False,
            with_end=False,
            with_padding=True,
            with_unknown=False)
        data_samples = [{}]
        postprocessor = CTCPostProcessor(max_seq_len=None, dictionary=dict_gen)

        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [1, 2, 100, 4, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8],
                                      [100, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 3, 100, 5, 6, 7, 8]]])
        data_samples = postprocessor(dummy_output, data_samples)
        self.assertEqual(data_samples[0]["pred_text"], '1020303')