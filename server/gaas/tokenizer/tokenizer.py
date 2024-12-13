import logging
from typing import List
from transformers import AutoTokenizer
from gaas.model_manager.model_files_manager import ModelFilesManager

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, max_tokens: int = 2048, **kwargs):
        self.model_files_path = ModelFilesManager().model_files_path
        self.tokenizer = self._get_tokenizer()
        self.max_tokens = max_tokens

    def _get_tokenizer(self) -> AutoTokenizer:
        try:
            return AutoTokenizer.from_pretrained(
                self.model_files_path, use_fast=False, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            logger.error(f"Full error:", exc_info=True)
            raise Exception("Failed to load tokenizer")

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
