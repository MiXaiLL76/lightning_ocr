from transformers import PreTrainedTokenizer
from typing import Optional, Tuple
import json
import os

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "mgp-str": "https://huggingface.co/alibaba-damo/mgp-str-base/blob/main/vocab.json",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mgp-str": 27}


class MgpstrTokenizer(PreTrainedTokenizer):
    """Construct a MGP-STR char tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        dict_list: list[str]:
            Characters list
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"[GO]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"[GO]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[s]"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, , defaults to `"[GO]"`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        dict_list=None,
        vocab_file=None,
        unk_token="[GO]",
        bos_token="[GO]",
        eos_token="[s]",
        pad_token="[GO]",
        **kwargs,
    ):
        assert (
            dict_list is not None or vocab_file is not None
        ), "dict_list or vocab_file must be provided"

        if dict_list is not None:
            base_bocab = {
                "[GO]": 0,
                "[s]": 1,
            }
            self.vocab = {
                char: (idx + len(base_bocab)) for idx, char in enumerate(dict_list)
            }
            self.vocab = dict(base_bocab, **self.vocab)
        else:
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.vocab = json.load(vocab_handle)

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.decoder = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string."""
        char_tokens = []
        for s in text:
            char_tokens.extend(s)
        return char_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise TypeError(
                "Vocabulary path ({}) should be a directory".format(save_directory)
            )

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )

        return (vocab_file,)
