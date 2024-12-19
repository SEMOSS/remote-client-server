from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import uuid
import logging
import time
from pydantic_models.request_models import ChatCompletionRequest
from model_utils.model_config import get_model_config
from pprint import pprint

logger = logging.getLogger(__name__)

chat_completion_router = APIRouter()


@chat_completion_router.post("/chat/completions")
async def chat_completions(request: Request):
    logger.info("Received chat completion request")

    try:
        request_data = await request.json()
        request_model = ChatCompletionRequest(**request_data)
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    # pprint(request_model)
    app = request.app
    queue_manager = app.state.queue_manager
    job_id = str(uuid.uuid4())

    model_type = get_model_config().get("type")

    # Don't stream vision models
    if request_model.stream and model_type == "text":

        async def event_stream():
            try:
                await queue_manager.add_job(job_id, request_model)
                logger.info(f"Added job {job_id} to queue")

                while True:
                    status = await queue_manager.get_job_status(job_id)
                    logger.info(f"Job {job_id} status: {status}")

                    if status == "queued":
                        queue_position = await queue_manager.get_job_position(job_id)
                        logger.info(f"Job {job_id} queue position: {queue_position}")
                        await asyncio.sleep(1)
                        continue

                    elif status == "complete":
                        chunks = await queue_manager.get_job_result(job_id)
                        logger.info(f"Job {job_id} completed with {len(chunks)} chunks")

                        if not isinstance(chunks, list):
                            logger.error(f"Expected list of chunks, got {type(chunks)}")
                            yield f"data: {json.dumps({'error': 'Invalid response format'})}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        # Send all chunks
                        for chunk in chunks:
                            yield f"data: {json.dumps(chunk)}\n\n"

                        # Send final [DONE] message
                        yield "data: [DONE]\n\n"
                        break

                    elif status in ["error", "cancelled", "timeout"]:
                        logger.error(f"Job {job_id} ended with status {status}")
                        yield f"data: {json.dumps({'error': f'Job {status}'})}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in event stream for job {job_id}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    # Handle non-streaming response
    else:
        try:
            await queue_manager.add_job(job_id, request_model)

            while True:
                status = await queue_manager.get_job_status(job_id)

                if status == "complete":
                    result = await queue_manager.get_job_result(job_id)
                    logger.info(f"Job {job_id} completed.")
                    return JSONResponse(content=result)

                elif status in ["error", "cancelled", "timeout"]:
                    error_response = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request_model.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"Error: Job {status}",
                                },
                                "finish_reason": status,
                            }
                        ],
                    }
                    return JSONResponse(status_code=500, content=error_response)

                await asyncio.sleep(1)  # Polling interval

        except Exception as e:
            logger.error(f"Error processing non-streaming request: {str(e)}")
            error_response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_model.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                        "finish_reason": "error",
                    }
                ],
            }
            return JSONResponse(status_code=500, content=error_response)
