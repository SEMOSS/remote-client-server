import os
import logging
from fastapi import APIRouter, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

queue_size_router = APIRouter()

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
QUEUE_SIZE_FILE = os.path.join(PROJECT_ROOT, "tmp", "queue_size.txt")


@queue_size_router.get("/queue_size")
async def get_queue_size():
    try:
        with open(QUEUE_SIZE_FILE, "r") as f:
            queue_size = int(f.read().strip())
        return {"queue_size": queue_size}
    except FileNotFoundError:
        logger.error(f"Queue size file not found: {QUEUE_SIZE_FILE}")
        return {"queue_size": 0}
    except ValueError:
        logger.error(f"Invalid queue size in file: {QUEUE_SIZE_FILE}")
        return {"queue_size": 0}
    except Exception as e:
        logger.error(f"Error reading queue size: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
