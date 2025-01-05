from typing import Optional, Sequence, Tuple, Dict

import torch
from lightning_ocr.models.postprocessors.base import BaseTextRecogPostprocessor


class AttentionPostprocessor(BaseTextRecogPostprocessor):
    """PostProcessor for seq2seq."""

    def get_single_prediction(
        self,
        probs: torch.Tensor,
        data_sample: Optional[Dict] = None,
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """Convert the output probabilities of a single image to index and
        score.

        Args:
            probs (torch.Tensor): Character probabilities with shape
                :math:`(T, C)`.
            data_sample (Dict): Datasample of an image. Defaults to None.

        Returns:
            tuple(list[int], list[float]): index and score.
        """
        max_value, max_idx = torch.max(probs, -1)
        index, score = [], []
        output_index = max_idx.cpu().detach().numpy().tolist()
        output_score = max_value.cpu().detach().numpy().tolist()
        for char_index, char_score in zip(output_index, output_score):
            if char_index in self.ignore_indexes:
                continue
            if char_index == self.dictionary.end_idx:
                break
            index.append(char_index)
            score.append(char_score)
        return index, score