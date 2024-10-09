from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
from model_utils.model_config import verify_payload
import logging

logger = logging.getLogger(__name__)

generation_router = APIRouter()


@generation_router.post("/generate")
async def http_generate(request: Request):
    logger.info(f"Received HTTP request: {request}")

    app = request.app
    queue_manager = app.state.queue_manager

    request_dict = await request.json()
    request_model = verify_payload(request_dict)

    job_id = str(uuid.uuid4())

    async def event_stream():
        await queue_manager.add_job(job_id, request_model.dict())

        while True:
            status = await queue_manager.get_job_status(job_id)
            if status == "queued":
                queue_position = await queue_manager.get_job_position(job_id)
                yield f"data: {json.dumps({'status': 'waiting', 'message': f'Your position in the queue is: {queue_position + 1}'})}\n\n"
            elif status == "processing":
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating...'})}\n\n"
            elif status == "complete":
                result = await queue_manager.get_job_result(job_id)
                result["status"] = "complete"
                result["message"] = "Generation complete."
                logger.info(f"Job {job_id} completed.")
                yield f"data: {json.dumps(result)}\n\n"
                break
            elif status in ["error", "cancelled", "timeout"]:
                yield f"data: {json.dumps({'status': status, 'message': f'Job {status}'})}\n\n"
                break

            await asyncio.sleep(1)  # Polling every second

    return StreamingResponse(event_stream(), media_type="text/event-stream")
