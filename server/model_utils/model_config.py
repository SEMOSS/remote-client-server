import os
import logging
from pydantic_models.request_models import (
    ImageRequest,
    TextRequest,
    GlinerRequest,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "pixart": {
        "model_repo_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "short_name": "pixart",
        "type": "image",
        "required_files": ["model_index.json"],
        "use_flash_attention": False,
    },
    "phi-3-mini-128k-instruct": {
        "model_repo_id": "microsoft/Phi-3-mini-128k-instruct",
        "short_name": "phi-3-mini-128k-instruct",
        "type": "text",
        "required_files": ["config.json", "generation_config.json"],
        "use_flash_attention": True,
    },
    "gliner-multi-v2-1": {
        "model_repo_id": "urchade/gliner_multi-v2.1",
        # NOTE: our kube deployments can't have '.' in the model name.. so we use '-' instead
        "short_name": "gliner-multi-v2-1",
        "type": "NER",
        "required_files": ["gliner_config.json"],
        "use_flash_attention": False,
    },
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


def get_repo_id() -> str:
    """
    Get the repo ID for the current model.
    Returns:
        str: Model repo ID
    """
    model_config = get_model_config()
    return model_config.get("model_repo_id")


def get_short_name() -> str:
    """
    Get the short name for the current model.
    Returns:
        str: Model short name
    """
    model_config = get_model_config()
    return model_config.get("short_name")


def get_model_type() -> str:
    """
    Get the type of the current model.
    Returns:
        str: Model type
    """
    model_config = get_model_config()
    return model_config.get("type", "")


def get_flash_attention() -> bool:
    """
    Get the flash attention availability for the current model.
    Returns:
        bool: Flash attention availability
    """
    model_config = get_model_config()
    return model_config.get("use_flash_attention", False)


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
    elif (
        model == "phi-3-mini-128k-instruct"
        or model == "microsoft/Phi-3-mini-128k-instruct"
    ):
        return TextRequest(**request)
    elif model == "gliner-multi-v2-1" or model == "urchade/gliner_multi-v2.1":
        return GlinerRequest(**request)
