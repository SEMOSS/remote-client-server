from gaas.text_gen.operations.instruct import Instruct
from gaas.text_gen.operations.chat import Chat
from gaas.text_gen.operations.json import JSON


class TextGenFactory:
    def __init__(self, model_name, **kwargs):
        self.instruct_gen = Instruct(model_name, **kwargs)
        self.chat_gen = Chat(model_name, **kwargs)
        self.json_gen = JSON(model_name, **kwargs)

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
