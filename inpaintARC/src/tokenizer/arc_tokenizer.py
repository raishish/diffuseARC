import os
import re
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class ARCTokenizer(PreTrainedTokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(self, vocab: Sequence[str], model_max_length: int, **kwargs):
        """
        Args:
            vocab (Sequence[str]): List of desired tokens.
            model_max_length (int): Model maximum sequence length.
        """
        self.vocab = vocab
        self.model_max_length = model_max_length

        # Define special tokens with AddedToken
        self.arc_sep_example_token = AddedToken("<arc_sep_example>", lstrip=False, rstrip=False, single_word=True)
        self.arc_sep_grid_token = AddedToken("<arc_sep_grid>", lstrip=False, rstrip=False, single_word=True)
        self.arc_sep_row_token = AddedToken("<arc_sep_row>", lstrip=False, rstrip=False, single_word=True)
        self.arc_grid_endx_token = AddedToken("<arc_grid_endx>", lstrip=False, rstrip=False, single_word=True)
        self.arc_grid_endy_token = AddedToken("<arc_grid_endy>", lstrip=False, rstrip=False, single_word=True)
        self.arc_grid_endxy_token = AddedToken("<arc_grid_endxy>", lstrip=False, rstrip=False, single_word=True)
        self.arc_pad_token = AddedToken("<arc_pad>", lstrip=False, rstrip=False, single_word=True)
        self.eos_token = AddedToken("<eos>", lstrip=False, rstrip=False, single_word=True)
        self.sep_token = AddedToken("<sep>", lstrip=False, rstrip=False, single_word=True)
        self.pad_token = AddedToken("<pad>", lstrip=False, rstrip=False, single_word=True)
        self.unk_token = AddedToken("<unk>", lstrip=False, rstrip=False, single_word=True)
        self.mask_token = AddedToken("<mask>", lstrip=True, rstrip=False, single_word=True)

        # Create vocab
        all_special_tokens = [
            str(self.arc_sep_example_token),
            str(self.arc_sep_grid_token),
            str(self.arc_sep_row_token),
            str(self.arc_grid_endx_token),
            str(self.arc_grid_endy_token),
            str(self.arc_grid_endxy_token),
            str(self.arc_pad_token),
            str(self.eos_token),
            str(self.sep_token),
            str(self.pad_token),
            str(self.unk_token),
            str(self.mask_token),
        ]

        self.vocab.extend(all_special_tokens)

        self._vocab_str_to_int = {}
        for id, token in enumerate(self.vocab):
            self._vocab_str_to_int[token] = id

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        # Regex matching for tokenization
        sorted_vocab_keys = sorted(self._vocab_str_to_int.keys(), key=len, reverse=True)
        pattern_str = "|".join(re.escape(token) for token in sorted_vocab_keys)
        # "|." at the end adds fallback for characters not matched by vocab tokens
        self._compiled_pattern = re.compile(r"(" + pattern_str + r"|.)")

        # Initialize with special tokens
        super().__init__(
            add_prefix_space=False,
            model_max_length=model_max_length,
            arc_sep_example_token=str(self.arc_sep_example_token),
            arc_sep_grid_token=str(self.arc_sep_grid_token),
            arc_sep_row_token=str(self.arc_sep_row_token),
            arc_grid_endx_token=str(self.arc_grid_endx_token),
            arc_grid_endy_token=str(self.arc_grid_endy_token),
            arc_grid_endxy_token=str(self.arc_grid_endxy_token),
            arc_pad_token=str(self.arc_pad_token),
            eos_token=str(self.eos_token),
            sep_token=str(self.sep_token),
            pad_token=str(self.pad_token),
            mask_token=str(self.mask_token),
            unk_token=str(self.unk_token),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        tokens = self._compiled_pattern.findall(text)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_config(self) -> Dict:
        return {
            "vocab": self.vocab,
            "model_max_length": self.model_max_length,
        }

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "ARCTokenizer":
        cfg = {}
        cfg["vocab"] = config['vocab']
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

        # Save special tokens (required by PreTrainedTokenizer)
        special_tokens_map = {
            "unk_token": str(self.unk_token),
            "sep_token": str(self.sep_token),
            "pad_token": str(self.pad_token),
            "mask_token": str(self.mask_token),
            "eos_token": str(self.eos_token),
            "additional_special_tokens": [
                str(self.arc_sep_example_token),
                str(self.arc_sep_grid_token),
                str(self.arc_sep_row_token),
                str(self.arc_grid_endx_token),
                str(self.arc_grid_endy_token),
                str(self.arc_grid_endxy_token),
                str(self.arc_pad_token),
            ],
        }
        special_tokens_map_file = Path(save_directory) / "special_tokens_map.json"
        with open(special_tokens_map_file, "w") as f:
            json.dump(special_tokens_map, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)

        # Load special tokens
        # special_tokens_map_file = Path(save_directory) / "special_tokens_map.json"
        # with open(special_tokens_map_file, "r") as f:
        #     special_tokens_map = json.load(f)
        #     kwargs.update(special_tokens_map)  # Pass special tokens to __init__

        # return cls.from_config(cfg, **kwargs)
        return cls.from_config(cfg)
