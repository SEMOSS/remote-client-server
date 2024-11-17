import logging
from gaas.text_gen.abstract_text_gen import AbstractTextGen

logger = logging.getLogger(__name__)


class Chat(AbstractTextGen):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def ask_model(
        self, prompt: str, temp: float = 0.1, prob: float = 0.2, max_tokens: int = 1024
    ):
        pass

    def generate(
        self,
        task: str,
        temp: float = 0.1,
        prob: float = 0.2,
        max_tokens: float = 1024,
        **kwargs,
    ):
        pass
