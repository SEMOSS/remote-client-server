import logging
from typing import List, Union
from transformers import AutoTokenizer, AutoProcessor
from gaas.model_manager.model_files_manager import ModelFilesManager
from model_utils.model_config import get_model_config

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, max_tokens: int = 2048, **kwargs):
        self.model_files_path = ModelFilesManager().get_model_files_path()
        self.model_config = get_model_config()
        self.tokenizer = self._get_tokenizer()
        self.max_tokens = max_tokens

    def _get_tokenizer(self) -> Union[AutoTokenizer, AutoProcessor]:
        try:
            model_type = self.model_config.get("type")

            if model_type == "vision":
                return AutoProcessor.from_pretrained(
                    self.model_files_path, trust_remote_code=True
                )
            else:
                return AutoTokenizer.from_pretrained(
                    self.model_files_path, use_fast=False, trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            logger.error(f"Full error:", exc_info=True)
            raise Exception(f"Failed to load tokenizer: {str(e)}")

    def count_tokens(self, input: Union[str, List[str]]) -> int:
        """Use the model tokenizer to get the number of tokens"""
        if isinstance(self.tokenizer, AutoProcessor):
            if isinstance(input, str):
                text_tokens = self.tokenizer.tokenizer(input)
                return len(text_tokens.input_ids[0])
            return 0  # image inputs
        else:
            input_tokens_ids = self.get_tokens_ids(input=input)
            return len(input_tokens_ids)

    def get_tokens_ids(
        self, input: Union[str, List[str]], add_special_tokens: bool = False
    ) -> List[int]:
        if isinstance(self.tokenizer, AutoProcessor):
            return self.tokenizer.tokenizer(
                input, add_special_tokens=add_special_tokens
            ).input_ids[0]
        return self.tokenizer.encode(input, add_special_tokens=add_special_tokens)

    def get_tokens(self, input: str) -> List[str]:
        if isinstance(self.tokenizer, AutoProcessor):
            return self.tokenizer.tokenizer.tokenize(input)
        return self.tokenizer.tokenize(input)

    def get_max_token_length(self) -> int:
        if self.max_tokens is None:
            if isinstance(self.tokenizer, AutoProcessor):
                return self.tokenizer.tokenizer.model_max_length
            return self.tokenizer.model_max_length
        return self.max_tokens

    def decode_token_ids(self, input: List[int]) -> str:
        if isinstance(self.tokenizer, AutoProcessor):
            return self.tokenizer.tokenizer.decode(input)
        return self.tokenizer.decode(input)
