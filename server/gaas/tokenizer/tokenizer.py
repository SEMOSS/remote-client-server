import os
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from model_utils.model_config import get_model_config


class Tokenizer:
    def __init__(self, max_tokens: int = 2048, **kwargs):
        self.tokenizer = self._get_tokenizer()
        self.max_tokens = max_tokens

    def _get_tokenizer(self) -> AutoTokenizer:
        model = get_model_config.get("model")
        # Used in development mode if you are running outside of a docker container otherwise model files should be loaded in attached volume
        model_files_local = os.environ.get("LOCAL_FILES") == "True"
        if model_files_local:
            # Get the absolute path of the project root directory
            script_dir = Path(__file__).resolve().parent  # gaas/text_gen
            project_root = script_dir.parent.parent.parent  # Go up to project root
            model_files_path = project_root / "model_files" / model
            # Convert to string and normalize
            model_files_path = str(model_files_path.resolve())
        else:
            model_files_path = f"/app/model_files/{model}"

        return AutoTokenizer.from_pretrained(model_files_path, use_fast=False)

    def count_tokens(self, input: str) -> int:
        """Use the model tokenizer to get the number of tokens"""
        input_tokens_ids = self.get_tokens_ids(input=input)
        return len(input_tokens_ids)

    def get_tokens_ids(self, input: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(input, add_special_tokens=add_special_tokens)

    def get_tokens(self, input: str) -> List[str]:
        return self.tokenizer.tokenize(input)

    def get_max_token_length(self) -> int:
        if self.max_tokens == None:
            return self.tokenizer.model_max_length
        else:
            return self.max_tokens

    def decode_token_ids(self, input: List[int]) -> str:
        return self.tokenizer.decode(input)
