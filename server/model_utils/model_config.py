import os
import logging
from pydantic_models.models import ImageRequest

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "pixart": {
        "model_repo_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "expected_model_class": "PixArtAlphaPipeline",
        "short_name": "pixart",
        "type": "image",
    }
}


def get_current_model() -> str:
    """
    Get the model that was used to start the server.
    Returns:
        str: Model name (short name)
    """
    model = os.getenv("MODEL", "")
    if model == "":
        logger.error("No model specified.")
        return "Model is not configured."
    return model.lower()


def get_model_config() -> dict:
    """
    Get the configuration for the current model.
    Returns:
        dict: Model configuration
    """
    model = get_current_model()
    model_config = SUPPORTED_MODELS.get(model, {})
    if not model_config:
        logger.error(f"Model {model} is not supported.")
        return None
    return model_config


def get_model_type() -> str:
    """
    Get the type of the current model.
    Returns:
        str: Model type
    """
    model_config = get_model_config()
    return model_config.get("type", "")


def get_short_name_from_request(request: dict) -> str:
    logger.info(f"Request: {request}")
    model = request.get("model", "")
    if not model:
        logger.error("The requested model is not specified.")
        return None
    # Check if the shortname was provided in the request
    if model in SUPPORTED_MODELS:
        return SUPPORTED_MODELS.get(model).get("short_name")
    # Check if the model repo ID was provided in the request
    for model_name, model_config in SUPPORTED_MODELS.items():
        if model_config.get("model_repo_id") == model:
            return model_config.get("short_name")
    logger.error(f"The requested model ({model}) is not supported.")
    return None


def verify_payload(request: dict):
    model = get_short_name_from_request(request)
    if model == None:
        logger.error("The payload verification failed.")
        return None
    elif model == "pixart" or model == "PixArt-alpha/PixArt-XL-2-1024-MS":
        return ImageRequest(**request)
