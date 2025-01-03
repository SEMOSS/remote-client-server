import logging
import torch
from PIL import Image
import base64
import io
from typing import List, Union
import requests
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import EmbeddingRequest, ImageInput
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

    def _process_image(self, image_data: Union[str, bytes, ImageInput]) -> Image.Image:
        """Process raw image data into PIL Image."""
        try:
            if isinstance(image_data, ImageInput):
                # Handle ImageInput object with URL
                image_bytes = self._download_image(image_data.image_url.url)
            elif isinstance(image_data, str):
                # Handle base64 encoded images
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")

    def _batch_process_images(
        self, images_data: List[Union[str, bytes, ImageInput]]
    ) -> torch.Tensor:
        """Process a batch of images into model input format."""
        processed_images = []
        for img_data in images_data:
            pil_image = self._process_image(img_data)
            processed = self.processor(images=pil_image, return_tensors="pt")
            processed_images.append(processed["pixel_values"])

        # Stack all processed images into a single batch
        return torch.cat(processed_images, dim=0).to(self.device)

    def generate(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            logger.info("Starting image embedding generation...")

            images_input = (
                [request.input]
                if isinstance(request.input, (str, bytes, ImageInput))
                else request.input
            )

            pixel_values = self._batch_process_images(images_input)

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
                "prompt_tokens": len(images_input),
                "total_tokens": len(images_input),
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
