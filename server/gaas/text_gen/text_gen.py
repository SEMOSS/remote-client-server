from gaas.text_gen.instruct import Instruct
from gaas.text_gen.chat import Chat


class TextGen:
    def __init__(self, model_name, **kwargs):
        self.instruct_gen = Instruct(model_name, **kwargs)
        self.chat_gen = Chat(model_name, **kwargs)

    def generate(self, request):
        operation = request.get("operation")
        if operation == "chat":
            return self.chat_gen.generate(**request)
        elif operation == "instruct":
            return self.instruct_gen.generate(**request)
        else:
            raise ValueError(f"Invalid operation type: {operation}")
