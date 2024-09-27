import logging
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

queue_router = APIRouter()


@queue_router.get("/queue", response_class=PlainTextResponse)
async def queue(request: Request):
    queue_manager = request.app.state.queue_manager
    queue_size = queue_manager.get_queue_size()
    return f"queue_size {queue_size}"
