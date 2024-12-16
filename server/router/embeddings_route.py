from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import uuid
import logging
import time
from pydantic_models.request_models import EmbeddingRequest
from pprint import pprint

logger = logging.getLogger(__name__)

embeddings_router = APIRouter()


@embeddings_router.post("/embeddings")
async def embeddings(request: Request):
    logger.info("Recieved embeddings request")

    try:
        request_data = await request.json()
        request_model = EmbeddingRequest(**request_data)
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    pprint(request_model)
    app = request.app
    queue_manager = app.state.queue_manager
    job_id = str(uuid.uuid4())

    try:
        await queue_manager.add_job(job_id, request_model)

        while True:
            status = await queue_manager.get_job_status(job_id)
            logger.info(f"Job {job_id} status: {status}")

            if status == "queued":
                queue_position = await queue_manager.get_job_position(job_id)
                logger.info(f"Job {job_id} queue position: {queue_position}")
                await asyncio.sleep(1)
                continue

            elif status == "complete":
                result = await queue_manager.get_job_result(job_id)
                logger.info(f"Job {job_id} completed")
                return JSONResponse(content=result)

            elif status in ["error", "cancelled", "timeout"]:
                logger.error(f"Job {job_id} failed with status: {status}")
                return JSONResponse(content={"status": status})

            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        return JSONResponse(content={"status": "error"})
