from typing import Dict, Union

import torch

from ...dictionary.dictionary import Dictionary
from .base import BaseTextRecogModuleLoss
from .ce_module_loss import CEModuleLoss


class ABIModuleLoss(BaseTextRecogModuleLoss):
    """Implementation of ABINet multiloss that allows mixing different types of
    losses with weights.

    Args:
        dictionary (:obj:`Dictionary`): The instance of `Dictionary`.
        max_seq_len (int): Maximum sequence length. The sequence is usually
            generated from decoder. Defaults to 40.
        letter_case (str): There are three options to alter the letter cases
            of gt texts:
            - unchanged: Do not change gt texts.
            - upper: Convert gt texts into uppercase characters.
            - lower: Convert gt texts into lowercase characters.
            Usually, it only works for English characters. Defaults to
            'unchanged'.
        weight_vis (float or int): The weight of vision decoder loss. Defaults
            to 1.0.
        weight_dec (float or int): The weight of language decoder loss.
            Defaults to 1.0.
        weight_fusion (float or int): The weight of fuser (aligner) loss.
            Defaults to 1.0.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        max_seq_len: int = 40,
        letter_case: str = "unchanged",
        weight_vis: Union[float, int] = 1.0,
        weight_lang: Union[float, int] = 1.0,
        weight_fusion: Union[float, int] = 1.0,
        **kwargs,
    ) -> None:
        assert isinstance(weight_vis, (float, int))
        assert isinstance(weight_lang, (float, int))
        assert isinstance(weight_fusion, (float, int))
        super().__init__(
            dictionary=dictionary, max_seq_len=max_seq_len, letter_case=letter_case
        )
        self.weight_vis = weight_vis
        self.weight_lang = weight_lang
        self.weight_fusion = weight_fusion
        self._ce_loss = CEModuleLoss(
            self.dictionary,
            max_seq_len,
            letter_case,
            reduction="mean",
            ignore_first_char=True,
        )

    def forward(self, outputs: torch.Tensor, data_samples: list[dict]) -> Dict:
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            data_samples (list[dict]): List are processed by ``get_target``.

        Returns:
            dict: A loss dictionary with ``loss_visual``, ``loss_lang`` and
            ``loss_fusion``. Each should either be the loss tensor or None if
            the output of its corresponding module is not given.
        """
        assert "out_vis" in outputs or "out_langs" in outputs or "out_fusers" in outputs
        losses = {}

        if outputs.get("out_vis", None):
            losses["loss_visual"] = (
                self.weight_vis
                * self._ce_loss(outputs["out_vis"]["logits"], data_samples)["loss_ce"]
            )

        if outputs.get("out_langs", None):
            lang_losses = []
            for out_lang in outputs["out_langs"]:
                lang_losses.append(
                    self._ce_loss(out_lang["logits"], data_samples)["loss_ce"]
                )
            losses["loss_lang"] = self.weight_lang * torch.mean(
                torch.stack(lang_losses)
            )

        if outputs.get("out_fusers", None):
            fuser_losses = []
            for out_fuser in outputs["out_fusers"]:
                fuser_losses.append(
                    self._ce_loss(out_fuser["logits"], data_samples)["loss_ce"]
                )
            losses["loss_fusion"] = self.weight_fusion * torch.mean(
                torch.stack(fuser_losses)
            )
        return losses
