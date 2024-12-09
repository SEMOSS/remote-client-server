import logging
import time
import uuid
import torch
from typing import Dict, Any, List, Optional, Union
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import ChatCompletionRequest
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]


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
            "max_new_tokens": min(request.max_tokens or 512, 512),
            "temperature": min(request.temperature or 0.3, 0.3),
            "top_p": min(request.top_p or 0.3, 0.3),
            "do_sample": True if request.temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
        }

    def _create_response_chunk(
        self, text: str, request, finish_reason: Optional[str] = None
    ) -> Dict:
        """Create a properly formatted response chunk."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _create_final_response(self, text: str, request, token_counts: Dict) -> Dict:
        """Create a properly formatted final response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": token_counts,
        }

    def generate(self, request: ChatCompletionRequest) -> Union[Dict, List[Dict]]:
        try:
            logger.info("Starting generation...")
            formatted_prompt = self._format_chat_prompt(request.messages)
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generation_config = self._prepare_generation_config(request)

            if request.stream:
                logger.info("Using streaming generation...")
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

                chunks = []
                for i in range(0, len(assistant_response), 4):
                    chunk_text = assistant_response[i : i + 4]
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    chunks.append(chunk)

                chunks.append(
                    {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": ""},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

                logger.info(f"Generated {len(chunks)} chunks")
                return chunks

            else:
                logger.info("Using non-streaming generation...")
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

                token_counts = {
                    "prompt_tokens": len(inputs["input_ids"][0]),
                    "completion_tokens": len(generated_ids[0])
                    - len(inputs["input_ids"][0]),
                    "total_tokens": len(generated_ids[0]),
                }

                return self._create_final_response(
                    assistant_response, request, token_counts
                )

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise
