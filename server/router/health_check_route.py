import logging
from fastapi import APIRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

health_check_router = APIRouter()


@health_check_router.get("/health")
async def health_check():
    logger.info("Health check successful")
    return {"status": "ok"}
