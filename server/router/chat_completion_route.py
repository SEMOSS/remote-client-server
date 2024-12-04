from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
import logging
from pydantic_models.request_models import ChatCompletionRequest

logger = logging.getLogger(__name__)

chat_completion_router = APIRouter()


@chat_completion_router.post("/chat/completions")
async def chat_completions(request: Request):
    logger.info(f"Received chat completion request")

    request_data = await request.json()
    # I can't put this into the params because I need to access app
    try:
        request_model = ChatCompletionRequest(**request_data)
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")

    app = request.app
    queue_manager = app.state.queue_manager

    job_id = str(uuid.uuid4())

    async def event_stream():
        await queue_manager.add_job(job_id, request_model)

        while True:
            status = await queue_manager.get_job_status(job_id)
            # if status == "queued":
            #     queue_position = await queue_manager.get_job_position(job_id)
            #     logger.info(
            #         f"data: {json.dumps({'status': 'waiting', 'message': f'Your position in the queue is: {queue_position + 1}'})}\n\n"
            #     )
            #     # yield f"data: {json.dumps({'status': 'waiting', 'message': f'Your position in the queue is: {queue_position + 1}'})}\n\n"
            # elif status == "processing":
            #     # yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating...'})}\n\n"
            #     logger.info(
            #         f"data: {json.dumps({'status': 'processing', 'message': 'Generating...'})}\n\n"
            #     )

            if status == "complete":
                result = await queue_manager.get_job_result(job_id)
                # result["status"] = "complete"
                # result["message"] = "Generation complete."
                logger.info(f"Job {job_id} completed.")
                yield f"{json.dumps(result)}\n\n"
                break
            elif status in ["error", "cancelled", "timeout"]:
                yield f"{json.dumps({'status': status, 'message': f'Job {status}'})}\n\n"
                break

            await asyncio.sleep(1)  # Polling every second

    return StreamingResponse(event_stream(), media_type="text/event-stream")
