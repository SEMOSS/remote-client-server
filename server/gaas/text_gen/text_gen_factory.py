from gaas.text_gen.operations.instruct import Instruct
from gaas.text_gen.operations.chat import Chat
from gaas.text_gen.operations.json import JSON
from gaas.model_manager.model_manager import ModelManager


class TextGenFactory:
    """Factory class that creates and manages different text generation operations."""

    def __init__(self, model_manager: ModelManager, **kwargs):
        # Initializing generation classes with shared model manager
        self.instruct_gen = Instruct(model_manager, **kwargs)
        self.chat_gen = Chat(model_manager, **kwargs)
        self.json_gen = JSON(model_manager, **kwargs)

    def generate(self, **kwargs):
        operation = kwargs.get("operation")
        if operation == "chat":
            return self.chat_gen.generate(**kwargs)
        elif operation == "instruct":
            return self.instruct_gen.generate(**kwargs)
        elif operation == "json":
            return self.json_gen.generate(**kwargs)
        else:
            raise ValueError(f"Invalid operation type: {operation}")
