from typing import List, Union
import numpy as np
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

space_key = "<SPACE>"


class FastTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        dict_list: List[str],
    ) -> None:
        self.start_token: str = "<BOS>"
        self.end_token: str = "<EOS>"
        self.padding_token: str = "<PAD>"
        self.unknown_token: str = "<UKN>"

        bpe_kwargs = {
            "vocab": {
                self.start_token: 0,
                self.padding_token: 1,
                self.end_token: 2,
                self.unknown_token: 3,
            },
            "merges": [],
            "unk_token": self.unknown_token,
        }

        for key in dict_list:
            if key in bpe_kwargs["vocab"]:
                continue

            if key == " ":
                key = space_key

            bpe_kwargs["vocab"][key] = len(bpe_kwargs["vocab"])

        tokenizer = Tokenizer(models.BPE(**bpe_kwargs))
        tokenizer.add_special_tokens(
            [self.start_token, self.end_token, self.padding_token, self.unknown_token]
        )
        tokenizer.enable_padding(
            pad_id=bpe_kwargs["vocab"][self.padding_token], pad_token=self.padding_token
        )

        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=self.start_token,
            eos_token=self.end_token,
            pad_token=self.padding_token,
            unk_token=self.unknown_token,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        decoded_str = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        return decoded_str.replace(" ", "").replace(space_key, " ")

    @property
    def __class__(self):
        return PreTrainedTokenizerFast
