import os
import logging
from pydantic_models.request_models import (
    ImageRequest,
    TextRequest,
    TextJSONRequest,
    NERRequest,
)

logger = logging.getLogger(__name__)

def get_model_config() -> dict:
    """
    Creates a model config by pulling the model values from the OS environment.
    """
    model = os.getenv("MODEL")
    model_repo_id = os.getenv("MODEL_REPO_ID")
    model_type = os.getenv("TYPE")
    return {
        "model": model,
        "model_repo_id": model_repo_id,
        "type": model_type,
    }

def verify_payload(request: dict):
    model_type = get_model_config().get("type")

    if model_type == None:
        logger.error("The payload verification failed.")
        return None
    elif model_type == "image":
        return ImageRequest(**request)
    elif model_type == "text":
        if "operation" in request and request["operation"] == "json":
            return TextJSONRequest(**request)
        else:
            logger.error("The requested operation is not supported.")
        # This is instruct request for now
        # return TextRequest(**request)
    elif model_type == "ner":
        return NERRequest(**request)
