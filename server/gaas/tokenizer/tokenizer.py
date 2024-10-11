import os
from typing import List
from transformers import AutoTokenizer
from model_utils.model_config import get_short_name


class Tokenizer:
    def __init__(self, max_tokens: int = 2048, **kwargs):
        self.tokenizer = self._get_tokenizer()
        self.max_tokens = max_tokens

    def _get_tokenizer(self) -> AutoTokenizer:
        short_name = get_short_name()
        # Used in development mode if you are running outside of a docker container otherwise model files should be loaded in attached volume
        model_files_local = os.environ.get("LOCAL_FILES") == "True"
        if model_files_local:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_files_path = os.path.join(
                script_dir, "..", "..", "model_files", short_name
            )
        else:
            model_files_path = f"/app/model_files/{short_name}"

        return AutoTokenizer.from_pretrained(model_files_path)

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
