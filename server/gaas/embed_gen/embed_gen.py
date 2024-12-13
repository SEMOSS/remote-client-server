import logging
import torch
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import EmbeddingRequest
from pydantic_models.response_models import EmbeddingResponse

logger = logging.getLogger(__name__)


class EmbedGen:
    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.device = model_manager.device

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit L2 norm."""
        embeddings = embeddings.to(self.device)
        normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
        return normalized

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform mean pooling on token embeddings using attention mask."""

        # Move tensors to the same device as the model
        token_embeddings = token_embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        result = sum_embeddings / sum_mask
        return result

    def generate(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            logger.info("Starting embedding generation...")

            inputs = (
                [request.input] if isinstance(request.input, str) else request.input
            )

            encoded_inputs = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192,
            )

            encoded_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in encoded_inputs.items()
            }

            with torch.no_grad():
                model_output = self.model(**encoded_inputs)

                token_embeddings = model_output.last_hidden_state

                # Mean pooling
                sentence_embeddings = self._mean_pooling(
                    token_embeddings, encoded_inputs["attention_mask"]
                )

                # Normalize embeddings
                normalized_embeddings = self._normalize_embeddings(sentence_embeddings)
                normalized_embeddings = normalized_embeddings.cpu()

            embeddings_list = normalized_embeddings.numpy().tolist()

            token_usage = {
                "prompt_tokens": sum(len(ids) for ids in encoded_inputs["input_ids"]),
                "total_tokens": sum(len(ids) for ids in encoded_inputs["input_ids"]),
            }

            embedding_data = [
                {"object": "embedding", "embedding": emb, "index": idx}
                for idx, emb in enumerate(embeddings_list)
            ]

            logger.info("Embedding generation completed successfully")
            return {"data": embedding_data, "usage": token_usage}

        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}", exc_info=True)
            raise
