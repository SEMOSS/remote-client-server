from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import asyncio
import uuid
from model_utils.model_config import verify_payload
import logging

logger = logging.getLogger(__name__)

sentiment_generation_router = APIRouter()


@sentiment_generation_router.post("/sentiment_generate")
async def generate_sentiment(request: Request):
    """
    API endpoint for sentiment analysis.
    Accepts a list of texts and returns sentiment predictions.
    """
    logger.info(f"Received Sentiment Generation Request")

    app = request.app
    queue_manager = app.state.queue_manager

    request_dict = await request.json()
    request_model = verify_payload(request_dict)

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

            await asyncio.sleep(1)  # Polling every second

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        return JSONResponse(content={"status": "error"})
