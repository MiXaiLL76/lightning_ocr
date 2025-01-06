from typing import List
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


class FastTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        dict_list: List[str],
    ) -> None:
        start_token: str = "<BOS>"
        end_token: str = "<EOS>"
        padding_token: str = "<PAD>"
        unknown_token: str = "<UKN>"

        bpe_kwargs = {
            "vocab": {
                start_token: 0,
                padding_token: 1,
                end_token: 2,
                unknown_token: 3,
            },
            "merges": [],
            "unk_token" : unknown_token,
        }

        for key in dict_list:
            if key in bpe_kwargs["vocab"]:
                continue

            bpe_kwargs["vocab"][key] = len(bpe_kwargs["vocab"])

        tokenizer = Tokenizer(models.BPE(**bpe_kwargs))
        tokenizer.add_special_tokens(
            [start_token, end_token, padding_token, unknown_token]
        )
        tokenizer.enable_padding(
            pad_id=bpe_kwargs["vocab"][padding_token], pad_token=padding_token
        )
        
        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=start_token,
            eos_token=end_token,
            pad_token=padding_token,
            unk_token=unknown_token
        )
