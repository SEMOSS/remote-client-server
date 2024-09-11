import logging
from fastapi import APIRouter, Path
from model_utils.download import verify_model_files

logger = logging.getLogger(__name__)

models_route = APIRouter()


@models_route.get("/models/{model}")
async def verify_models(
    model: str = Path(..., description="The name of the model to verify")
):
    model_files = verify_model_files(model)
    return {"message": model_files}
