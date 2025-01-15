import logging
import torch
from PIL import Image
import io
from typing import List
import requests
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

    def _process_image(self, image_url: str) -> Image.Image:
        """Process image from URL into PIL Image."""
        try:
            image_bytes = self._download_image(image_url)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")

    def _batch_process_images(self, image_urls: List[str]) -> torch.Tensor:
        """Process a batch of image URLs into model input format."""
        processed_images = []
        for url in image_urls:
            pil_image = self._process_image(url)
            processed = self.processor(images=pil_image, return_tensors="pt")
            processed_images.append(processed["pixel_values"])

        return torch.cat(processed_images, dim=0).to(self.device)

    def generate(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            logger.info("Starting image embedding generation...")

            image_urls = request.get_image_urls()

            pixel_values = self._batch_process_images(image_urls)

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
                "prompt_tokens": len(image_urls),
                "total_tokens": len(image_urls),
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
