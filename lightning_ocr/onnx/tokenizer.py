from typing import Dict, List, Union, Optional
import numpy as np
import json

class OnnxTokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
        eos_token : Optional[str] = None,
        bos_token : Optional[str] = None,
    ) -> None:
        self.vocab = vocab
        assert len(self.vocab), "vocab is empty"
        self.special_tokens = set(special_tokens or [])
        
        self.eos_token = eos_token
        self.eos_token_id = self.vocab.get(self.eos_token, -1)
        
        self.bos_token = bos_token
        self.bos_token_id = self.vocab.get(self.bos_token, -1)
        
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    @classmethod
    def load_from_file(cls, vocab : str, special_tokens_map : str):
        with open(vocab, "r") as f:
            vocab = json.load(f)
        
        with open(special_tokens_map, "r") as f:
            special_tokens = json.load(f)
        
        kwargs = {
            "vocab" : vocab,
            "special_tokens" : special_tokens.keys(),
            "eos_token" : special_tokens.get("eos_token", None),
            "bos_token" : special_tokens.get("bos_token", None)
        }
        
        if kwargs["bos_token"] is None:
            kwargs["bos_token"] = special_tokens.get("cls_token", None)
        
        return cls(**kwargs)
    
    def decode_line(self, token_ids : "np.ndarray", skip_special_tokens: bool = False, join_tokens : bool = False):
        line = []
        for token in token_ids:
            _char = self.idx_to_char[token]
            if skip_special_tokens:
                if token in self.special_tokens:
                    continue
            line.append(_char)
        
        if join_tokens:
            line = "".join(line)
        
        return line

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray"],
        skip_special_tokens: bool = False, 
        join_tokens : bool = False
    ) -> str:
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, list):
            token_ids = token_ids
        elif isinstance(token_ids, np.ndarray):
            pass
        else:
            raise ValueError(f"token_ids must be int, list or np.ndarray, but got {type(token_ids)}")
        
        token_ids = np.array(token_ids)
        if len(token_ids.shape) == 1:
            return self.decode_line(token_ids, skip_special_tokens)
        else:
            return [self.decode_line(line, skip_special_tokens) for line in token_ids]