import logging
from fastapi import APIRouter, HTTPException
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gpu_router = APIRouter()


@gpu_router.get("/gpu")
async def is_gpu_available():
    is_gpu = torch.cuda.is_available()
    if is_gpu is None:
        raise HTTPException(
            status_code=500, detail="Could not obtain GPU availability."
        )
    return {"gpu": is_gpu}
