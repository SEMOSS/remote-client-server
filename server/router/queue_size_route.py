import logging
from fastapi import APIRouter, HTTPException
from sockets.queue_manager import queue_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

queue_size_router = APIRouter()


@queue_size_router.get("/queue_size")
async def get_queue_size():
    size = queue_manager.get_queue_size()
    if size is None:
        raise HTTPException(status_code=500, detail="Queue size not available")
    return {"queue_size": size}
