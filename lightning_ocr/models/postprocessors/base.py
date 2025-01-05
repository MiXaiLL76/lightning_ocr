import warnings
from typing import Sequence, Tuple

import torch
from lightning_ocr.dictionary.dictionary import Dictionary


class BaseTextRecogPostprocessor:
    """Base text recognition postprocessor.

    Args:
        dictionary ( :obj:`Dictionary`): The the instance of `Dictionary`.
        max_seq_len (int): max_seq_len (int): Maximum sequence length. The
            sequence is usually generated from decoder. Defaults to 40.
        ignore_chars (list[str]): A list of characters to be ignored from the
            final results. Postprocessor will skip over these characters when
            converting raw indexes to characters. Apart from single characters,
            each item can be one of the following reversed keywords: 'padding',
            'end' and 'unknown', which refer to their corresponding special
            tokens in the dictionary.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        max_seq_len: int = 40,
        ignore_chars: Sequence[str] = ["padding"],
        **kwargs,
    ) -> None:
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

        mapping_table = {
            "padding": self.dictionary.padding_idx,
            "end": self.dictionary.end_idx,
            "unknown": self.dictionary.unknown_idx,
        }

        ignore_indexes = list()
        for ignore_char in ignore_chars:
            assert isinstance(ignore_char, str), "ignore_chars must be list of str"

            index = mapping_table.get(
                ignore_char, self.dictionary.char2idx(ignore_char, strict=False)
            )
            if index is None or (
                index == self.dictionary.unknown_idx and ignore_char != "unknown"
            ):
                warnings.warn(
                    f"{ignore_char} does not exist in the dictionary", UserWarning
                )
                continue
            ignore_indexes.append(index)
        self.ignore_indexes = ignore_indexes

    def get_single_prediction(
        self, probs: torch.Tensor, data_sample: dict = None
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output probabilities of a single image to index and
        score.

        Args:
           probs (torch.Tensor): Character probabilities with shape
                :math:`(T, C)`.
           data_sample (Dict): Datasample of an image.

        Returns:
            tuple(list[int], list[float]): Index and scores per-character.
        """
        raise NotImplementedError

    def __call__(
        self, probs: torch.Tensor, data_samples: Sequence[dict]
    ) -> Sequence[dict]:
        """Convert outputs to strings and scores.

        Args:
            probs (torch.Tensor): Batched character probabilities, the model's
                softmaxed output in size: :math:`(N, T, C)`.
            data_samples (list[dict]): The list of dict

        Returns:
            list(dict): The list of dict. It usually contain ``pred_text`` information.
        """
        batch_size = probs.size(0)

        for idx in range(batch_size):
            index, score = self.get_single_prediction(
                probs[idx, :, :], data_samples[idx]
            )
            text = self.dictionary.idx2str(index)
            data_samples[idx]["pred_text"] = text
            data_samples[idx]["pred_score"] = score

        return data_samples
