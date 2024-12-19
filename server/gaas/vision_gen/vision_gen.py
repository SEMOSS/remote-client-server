import logging
import torch
from typing import List, Dict, Any
from PIL import Image
import base64
import io
import time
import uuid
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import ChatCompletionRequest

logger = logging.getLogger(__name__)


class VisionGen:
    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.processor = model_manager.processor
        self.tokenizer = model_manager.tokenizer
        self.device = model_manager.device

    def _prepare_generation_config(
        self, request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        max_new_tokens = request.max_new_tokens or request.max_tokens or 1024
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": min(request.temperature, 0.3),
            "top_p": min(request.top_p, 0.3),
            "do_sample": True if request.temperature > 0 else False,
            "num_beams": 5,
            "length_penalty": 1.0,
            "repetition_penalty": 1.2,
        }

    def _validate_base64_image(self, image_data: str) -> bool:
        """Try to decode and open image to verify its validity."""
        try:
            img_bytes = base64.b64decode(image_data)
            Image.open(io.BytesIO(img_bytes))
            return True
        except Exception as e:
            logger.error(f"Invalid base64 image data: {str(e)}")
            return False

    def _process_batch(
        self, images: List[str], prompt: str = ""
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of images with optional prompt."""
        if prompt:
            inputs = self.processor(
                images=[
                    Image.open(io.BytesIO(base64.b64decode(img))) for img in images
                ],
                text=prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                images=[
                    Image.open(io.BytesIO(base64.b64decode(img))) for img in images
                ],
                return_tensors="pt",
            )

        float16_keys = {"pixel_values", "image_embeds", "image_features"}
        long_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}

        # Move tensors to device with appropriate dtype
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k in float16_keys:
                    processed_inputs[k] = v.to(device=self.device, dtype=torch.float16)
                elif k in long_keys:
                    processed_inputs[k] = v.to(device=self.device, dtype=torch.long)
                else:
                    processed_inputs[k] = v.to(device=self.device)
            else:
                processed_inputs[k] = v

        return processed_inputs

    def generate(self, request: ChatCompletionRequest) -> Dict:
        """Generate vision model outputs."""
        try:
            logger.info("Starting vision generation...")

            message = request.messages[-1]  # Get last message
            image_data = None
            prompt = ""

            if isinstance(message.content, list):
                for content in message.content:
                    if content.type == "image_url" and content.image_url is not None:
                        image_url = content.image_url["url"]
                        # Strip data URL prefix if present
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                        else:
                            image_data = image_url

                        if not self._validate_base64_image(image_data):
                            raise ValueError("Invalid image data provided")

                    elif content.type == "text" and content.text is not None:
                        prompt = content.text
            elif isinstance(message.content, str):
                prompt = message.content

            if image_data is None:
                raise ValueError("No valid image data found in the request")

            inputs = self._process_batch([image_data], prompt)
            generation_config = self._prepare_generation_config(request)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            generated_text = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            input_token_length = (
                len(inputs.get("input_ids", [0])[0]) if "input_ids" in inputs else 0
            )
            token_usage = {
                "prompt_tokens": input_token_length,
                "completion_tokens": outputs.shape[1],
                "total_tokens": input_token_length + outputs.shape[1],
            }

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": token_usage,
            }

        except Exception as e:
            logger.error(f"Error during vision generation: {str(e)}", exc_info=True)
            raise

    async def generate_stream(self, request: ChatCompletionRequest):
        """Stream generation results."""
        pass
