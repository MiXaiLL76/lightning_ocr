from unittest import TestCase, mock

import torch
from lightning_ocr.dictionary.dictionary import Dictionary
from lightning_ocr.models.postprocessors.base import BaseTextRecogPostprocessor

class TestBaseTextRecogPostprocessor(TestCase):

    def test_init(self):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        # test diction cfg
        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        
        base_postprocessor = BaseTextRecogPostprocessor(dict_cfg)
        self.assertIsInstance(base_postprocessor.dictionary, Dictionary)
        self.assertListEqual(base_postprocessor.ignore_indexes,
                             [base_postprocessor.dictionary.padding_idx])

        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])

        self.assertListEqual(base_postprocessor.ignore_indexes, [1, 2, 3])

        # test ignore_chars
        with self.assertRaisesRegex(AssertionError,
                                    'ignore_chars must be list of str'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=[1, 2, 3])
        with self.assertWarnsRegex(Warning,
                                   'M does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=['M'])

        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])

        # test diction cfg with with_unknown=False
        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=False)
        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])

        self.assertListEqual(base_postprocessor.ignore_indexes, [1, 2, 3])

        # test ignore_chars
        with self.assertRaisesRegex(AssertionError,
                                    'ignore_chars must be list of str'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=[1, 2, 3])

        with self.assertWarnsRegex(Warning,
                                   'M does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                dict_cfg, ignore_chars=['M'])

        with self.assertWarnsRegex(Warning,
                                   'M does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                Dictionary(
                    chars,
                    with_unknown=True,
                    unknown_token=None),
                ignore_chars=['M'])

        with self.assertWarnsRegex(Warning,
                                   'M does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                Dictionary(
                    chars, with_unknown=True),
                ignore_chars=['M'])

        with self.assertWarnsRegex(Warning,
                                   'unknown does not exist in the dictionary'):
            base_postprocessor = BaseTextRecogPostprocessor(
                Dictionary(
                    chars,
                    with_unknown=False),
                ignore_chars=['unknown'])

        base_postprocessor = BaseTextRecogPostprocessor(
            dict_cfg, ignore_chars=['1', '2', '3'])

    @mock.patch(f'{__name__}.BaseTextRecogPostprocessor.get_single_prediction')
    def test_call(self, mock_get_single_prediction):

        def mock_func(output, data_sample):
            return [0, 1, 2], [0.8, 0.7, 0.9]

        chars = "0123456789abcdefghijklmnopqrstuvwxyz"

        dict_cfg = Dictionary(
            chars,
            with_start=True,
            with_end=True,
            same_start_end=False,
            with_padding=True,
            with_unknown=True)
        mock_get_single_prediction.side_effect = mock_func
        data_samples = [{"gt_text" : "012"}]
        postprocessor = BaseTextRecogPostprocessor(
            max_seq_len=None, dictionary=dict_cfg)

        # test decode output to index
        dummy_output = torch.Tensor([[[1, 100, 3, 4, 5, 6, 7, 8]]])
        data_samples = postprocessor(dummy_output, data_samples)
        self.assertEqual(data_samples[0]["pred_text"], '012')
