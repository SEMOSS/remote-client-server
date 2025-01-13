from fastapi import APIRouter
from gaas.model_manager.model_manager import ModelManager

model_loaded_check_router = APIRouter()


@model_loaded_check_router.get("/model-loaded", response_model=bool)
async def is_model_loaded():
    """Check if the model has finished loading."""
    model_manager = ModelManager.get_instance()
    return model_manager._initialized
