import logging
from fastapi import APIRouter, Header, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

health_check_router = APIRouter()


@health_check_router.get("/health_check")
async def health_check():
    logger.info("Health check successful")
    return {"status": "ok"}
