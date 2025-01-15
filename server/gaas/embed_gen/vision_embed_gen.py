import logging
import torch
from PIL import Image
import io
from typing import List
import requests
import base64
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import EmbeddingRequest
from pydantic_models.response_models import EmbeddingResponse

logger = logging.getLogger(__name__)


class VisionEmbedGen:
    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.processor = model_manager.processor
        self.device = model_manager.device

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit L2 norm."""
        embeddings = embeddings.to(self.device)
        normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
        return normalized

    def _download_image(self, url: str) -> bytes:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            raise ValueError(f"Failed to download image from URL: {str(e)}")

    def _decode_base64(self, base64_string: str) -> bytes:
        """Decode base64 image string to bytes."""
        try:
            if base64_string.startswith("data:image/"):
                base64_string = base64_string.split(",")[1]
            elif base64_string.startswith("base64:"):
                base64_string = base64_string[7:]

            return base64.b64decode(base64_string)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise ValueError(f"Invalid base64 image data: {str(e)}")

    def _process_image_input(self, image_input: str) -> Image.Image:
        """Process image from URL or base64 into PIL Image."""
        try:
            if image_input.startswith(("http://", "https://")):
                image_bytes = self._download_image(image_input)
            else:
                image_bytes = self._decode_base64(image_input)

            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")

    def _batch_process_images(self, image_inputs: List[str]) -> torch.Tensor:
        """Process a batch of image inputs into model input format."""
        processed_images = []
        for input_data in image_inputs:
            pil_image = self._process_image_input(input_data)
            processed = self.processor(images=pil_image, return_tensors="pt")
            processed_images.append(processed["pixel_values"])

        return torch.cat(processed_images, dim=0).to(self.device)

    def generate(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            logger.info("Starting image embedding generation...")

            image_inputs = request.get_image_inputs()
            pixel_values = self._batch_process_images(image_inputs)

            with torch.no_grad():
                outputs = self.model(pixel_values)

                if hasattr(outputs, "image_embeds"):
                    embeddings = outputs.image_embeds
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                normalized_embeddings = self._normalize_embeddings(embeddings)
                normalized_embeddings = normalized_embeddings.cpu()

            embeddings_list = normalized_embeddings.numpy().tolist()

            token_usage = {
                "prompt_tokens": len(image_inputs),
                "total_tokens": len(image_inputs),
            }

            embedding_data = [
                {"object": "embedding", "embedding": emb, "index": idx}
                for idx, emb in enumerate(embeddings_list)
            ]

            logger.info("Image embedding generation completed successfully")
            return {"data": embedding_data, "usage": token_usage}

        except Exception as e:
            logger.error(
                f"Error during image embedding generation: {str(e)}", exc_info=True
            )
            raise
