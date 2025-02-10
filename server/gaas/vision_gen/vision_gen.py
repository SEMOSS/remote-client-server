import gc
import logging
import torch
from typing import List, Dict, Any
from PIL import Image
import requests
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
        """Handle Qwen2VL's special token requirements"""
        max_new_tokens = request.max_new_tokens or request.max_tokens or 1024
        config = {
            "max_new_tokens": max_new_tokens,
            "temperature": min(max(request.temperature, 0.01), 1.0),
            "top_p": min(max(request.top_p, 0.01), 1.0),
            "do_sample": True,
            "num_beams": 1,
        }

        tokenizer = getattr(self.processor, "tokenizer", self.tokenizer)
        if tokenizer is not None:
            if hasattr(tokenizer, "eos_token_id"):
                config["eos_token_id"] = tokenizer.eos_token_id
            if hasattr(tokenizer, "pad_token_id"):
                config["pad_token_id"] = tokenizer.pad_token_id

        return config

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
        """Process a batch of images with proper image token insertion"""
        pil_images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]

        model_type = getattr(self.model.config, "model_type", "unknown")

        if model_type in ["qwen2_vl", "qwen2-vl"]:

            image_token_id = self.model.config.image_token_id
            logger.info(f"Using image token ID: {image_token_id}")

            image_token = getattr(self.processor, "image_token", "<image>")
            formatted_text = f"User: {image_token}{prompt}\nAssistant:"

            inputs = self.processor(
                text=formatted_text,
                images=pil_images[0],
                return_tensors="pt",
                padding=True,
            )

            image_token_id = self.model.config.image_token_id
            num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()
            logger.info(f"Image token count after processing: {num_image_tokens}")
            logger.info(f"Input shape after processing: {inputs['input_ids'].shape}")

        else:
            # Non Qwen2VL models for now..
            inputs = self.processor(
                images=pil_images, text=prompt, return_tensors="pt", padding=True
            )

        logger.info("Processor output structure:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"Input '{k}' shape: {v.shape}, dtype: {v.dtype}")
            else:
                logger.info(f"Input '{k}' type: {type(v)}")

        # Move tensors to device with appropriate dtypes
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k in {"pixel_values", "image_embeds", "image_features"}:
                    processed_inputs[k] = v.to(device=self.device, dtype=torch.float16)
                elif k in {
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "position_ids",
                }:
                    processed_inputs[k] = v.to(device=self.device, dtype=torch.long)
                else:
                    processed_inputs[k] = v.to(device=self.device)
            else:
                processed_inputs[k] = v

        return processed_inputs

    def url_to_base64(self, image_url: str) -> str:
        """Convert image URL to base64 string with proper error handling.

        Args:
            image_url: URL of the image to fetch

        Returns:
            Base64 encoded string of the image

        Raises:
            ValueError: If image cannot be fetched or processed
        """
        try:
            response = requests.get(image_url, timeout=10)  # Add timeout
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"Invalid content type: {content_type}")

            try:
                Image.open(io.BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Invalid image data: {str(e)}")

            base64_string = base64.b64encode(response.content).decode("utf-8")
            return base64_string

        except requests.exceptions.Timeout:
            raise ValueError("Timeout while fetching image")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def generate(self, request: ChatCompletionRequest) -> Dict:
        """Generate vision model outputs."""
        logger = logging.getLogger(__name__)
        try:
            logger.info("Starting vision generation...")

            # Log model configuration
            logger.info(f"Model type: {self.model.config.model_type}")
            logger.info(
                f"Model architecture: {self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') else 'unknown'}"
            )

            message = request.messages[-1]
            image_data = None
            prompt = ""

            if isinstance(message.content, list):
                for content in message.content:
                    if content.type == "image_url" and content.image_url is not None:
                        image_url = content.image_url["url"]
                        logger.info(f"Processing image URL: {image_url}")

                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            logger.info("Extracted base64 data from data URL")
                        else:
                            image_data = self.url_to_base64(image_url)
                            logger.info("Converted URL to base64")

                        if not self._validate_base64_image(image_data):
                            raise ValueError("Invalid image data provided")
                        logger.info("Image validation successful")

                    elif content.type == "text" and content.text is not None:
                        prompt = content.text
                        logger.info(f"Found text prompt: {prompt}")
            elif isinstance(message.content, str):
                prompt = message.content
                logger.info(f"Using string content as prompt: {prompt}")

            if image_data is None:
                raise ValueError("No valid image data found in the request")

            logger.info("Processing inputs...")
            inputs = self._process_batch([image_data], prompt)
            generation_config = self._prepare_generation_config(request)
            logger.info(f"Generation config: {generation_config}")

            with torch.no_grad():
                logger.info(
                    f"Pre-generation memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB"
                )

                generate_inputs = {**inputs, **generation_config}
                logger.info("Generate inputs:")
                for k, v in generate_inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"  {k}: tensor of shape {v.shape}")
                    else:
                        logger.info(f"  {k}: {v}")

                outputs = self.model.generate(**inputs, **generation_config)
                logger.info(
                    f"Post-generation memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB"
                )
                logger.info(f"Output shape: {outputs.shape}")

            # Testing some manual memory management
            del inputs
            torch.cuda.empty_cache()

            generated_text = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            logger.info(f"Generated text: {generated_text}")

            input_token_length = (
                len(inputs.get("input_ids", [0])[0]) if "input_ids" in inputs else 0
            )
            token_usage = {
                "prompt_tokens": input_token_length,
                "completion_tokens": outputs.shape[1],
                "total_tokens": input_token_length + outputs.shape[1],
            }
            logger.info(f"Token usage: {token_usage}")

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

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    async def generate_stream(self, request: ChatCompletionRequest) -> Dict:
        """Handle streaming generation for vision models.
        For vision models, we generate the full response first and then format it as a stream.
        """
        try:
            response = self.generate(request)
            return response

        except Exception as e:
            logger.error(
                f"Error during vision generation streaming: {str(e)}", exc_info=True
            )
            raise
