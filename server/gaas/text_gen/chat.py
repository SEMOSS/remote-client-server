import logging
import time
import uuid
import torch
from typing import Dict, Any
from transformers import TextIteratorStreamer
from threading import Thread
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import ChatCompletionRequest

logger = logging.getLogger(__name__)


class Chat:
    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.pipe = model_manager.pipe
        self.device = model_manager.device

    def _format_chat_prompt(self, messages: list) -> str:
        """Format the chat messages into a single prompt string."""
        formatted_prompt = ""
        for message in messages:
            role = message.role
            content = message.content

            if role == "system":
                formatted_prompt += f"System: {content}\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n"

        formatted_prompt += "Assistant: "
        return formatted_prompt

    def _prepare_generation_config(self, request) -> Dict[str, Any]:
        """Prepare the generation configuration based on the request parameters."""
        return {
            "max_new_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": True if request.temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.0,
        }

    def generate(self, request: ChatCompletionRequest):
        try:
            formatted_prompt = self._format_chat_prompt(request.messages)
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generation_config = self._prepare_generation_config(request)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config,
                )

            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            assistant_response = generated_text[len(formatted_prompt) :].strip()
            input_token_count = len(inputs["input_ids"][0])
            output_token_count = len(generated_ids[0]) - input_token_count

            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count,
                },
            }

            return response

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

        # async def generate_stream(self, request: ChatCompletionRequest):
        #     """Generate a streaming response for the chat completion request."""
        #     try:
        #         # Format the prompt from the messages
        #         formatted_prompt = self._format_chat_prompt(request.messages)

        #         # Tokenize the input
        #         inputs = self.tokenizer(
        #             formatted_prompt,
        #             return_tensors="pt",
        #             padding=True,
        #             truncation=True,
        #             max_length=2048,
        #         ).to(self.device)

        #         # Prepare generation config
        #         generation_config = self._prepare_generation_config(request)

        #         # Setup the streamer
        #         streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        #         # Create generation kwargs
        #         generation_kwargs = {
        #             "input_ids": inputs["input_ids"],
        #             "attention_mask": inputs["attention_mask"],
        #             "streamer": streamer,
        #             **generation_config,
        #         }

        #         # Start generation in a separate thread
        #         thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        #         thread.start()

        #         # Stream the output
        #         generated_text = ""
        #         for new_text in streamer:
        #             generated_text += new_text
        #             yield {
        #                 "choices": [
        #                     {
        #                         "index": 0,
        #                         "delta": {"role": "assistant", "content": new_text},
        #                         "finish_reason": None,
        #                     }
        #                 ]
        #             }

        #         # Send the final message
        #         yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

        # except Exception as e:
        #     logger.error(f"Error during streaming generation: {str(e)}")
        #     raise
