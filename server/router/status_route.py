import logging
import torch
from fastapi import APIRouter
from globals.globals import get_server_status
from sockets.queue_manager import queue_manager
from model_utils.model_config import get_model_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

status_route = APIRouter()


def is_gpu_available() -> bool:
    """
    Checks if the container or your local environment is GPU-enabled.
    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()


def get_queue_size() -> int:
    """
    Get the size of the queue. Returns 0 if the queue manager hasn't been initialized.
    Returns:
        int: Size of the queue
    """
    # if the QueueManager hasn't been initialized (i.e. downloading model files)
    if not queue_manager or not queue_manager.get_queue_size():
        return 0
    return queue_manager.get_queue_size()


def get_status() -> str:
    """
    Get the status of the server. initializing, DOWNdownloading_model, or ready.
    Returns:
        str: Server status
    """
    return get_server_status().value


def get_model() -> dict:
    """
    Get the model configuration.
    Returns:
        dict: Model configuration with the following structure:
            {
                "model_repo_id": str,
                "expected_model_class": str,
                "type": str
            }

    """
    model_config = get_model_config()
    return model_config


@status_route.get("/status")
async def status():
    model_config = get_model()
    model = model_config.get("model_repo_id")
    type = model_config.get("type")
    return {
        "status": get_status(),
        "gpu": is_gpu_available(),
        "queue": get_queue_size(),
        "model": model,
        "type": type,
    }
