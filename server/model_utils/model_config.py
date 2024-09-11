import os
import logging

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
        str: Model name
    """
    model = os.getenv("MODEL", "pixart")
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
