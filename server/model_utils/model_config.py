import os
import logging
from pydantic_models.request_models import (
    ImageRequest,
    NERRequest,
    EmbeddingRequest,
)

logger = logging.getLogger(__name__)


def get_model_config() -> dict:
    """
    Creates a model config by pulling the model values from the OS environment.
    """
    model = os.getenv("MODEL")
    model_repo_id = os.getenv("MODEL_REPO_ID")
    model_type = os.getenv("MODEL_TYPE")
    semoss_id = os.getenv("SEMOSS_ID", "N/A")
    use_vllm = os.getenv("USE_VLLM", "false").lower() in ("true", "1", "yes")

    if model_type == "text":
        use_vllm = True

    return {
        "model": model.lower(),
        "model_repo_id": model_repo_id,
        "type": model_type.lower(),
        "semoss_id": semoss_id,
        "use_vllm": use_vllm,
    }


def verify_payload(request: dict):
    model_config = get_model_config()
    model_type = model_config.get("type")

    if model_type == None:
        logger.error("The payload verification failed.")
        return None
    elif model_type == "image":
        return ImageRequest(**request)
    elif model_type == "embed":
        return EmbeddingRequest(**request)
    elif model_type == "ner":
        return NERRequest(**request)
