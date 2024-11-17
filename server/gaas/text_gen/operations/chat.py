import logging
from gaas.model_manager.model_manager import ModelManager

logger = logging.getLogger(__name__)


class Chat:
    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.pipe = model_manager.pipe
        self.device = model_manager.device

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
