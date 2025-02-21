from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import uuid
import logging
import time
from pydantic_models.request_models import ChatCompletionRequest
from model_utils.model_config import get_model_config

logger = logging.getLogger(__name__)
chat_completion_router = APIRouter()


def format_vllm_response(output, request_model):
    """Format vLLM output to match OpenAI API format"""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output.outputs[0].text},
                "finish_reason": output.outputs[0].finish_reason,
            }
        ],
    }


def format_vllm_stream_chunk(text, finish_reason, request_model):
    """Format streaming chunk for vLLM output"""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_model.model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def format_vision_stream_chunk(text, finish_reason, request_model, include_role=False):
    """Format streaming chunk for vision model output"""
    delta = {"content": text} if text else {}
    if include_role:
        delta["role"] = "assistant"

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_model.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def format_messages_prompt(messages):
    """Convert chat messages to prompt format"""
    formatted = []
    for msg in messages:
        try:
            formatted.append(f"{msg.role.capitalize()}: {msg.content}")
        except Exception as e:
            logger.error(f"Error formatting message: {str(e)}")
            raise ValueError(f"Failed to format message: {str(e)}")

    if not formatted:
        raise ValueError("No messages to format")

    return "\n".join(formatted)


@chat_completion_router.post("/chat/completions")
async def chat_completions(request: Request):
    logger.info("Received chat completion request")

    try:
        request_data = await request.json()
        request_model = ChatCompletionRequest(**request_data)
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    model_config = get_model_config()
    model_type = model_config.get("type")
    use_vllm = model_config.get("use_vllm", False)

    if model_type == "vision":
        app = request.app
        queue_manager = app.state.queue_manager
        job_id = str(uuid.uuid4())

        if request_model.stream:

            async def vision_stream():
                try:
                    await queue_manager.add_job(job_id, request_model)
                    logger.info(f"Added vision job {job_id} to queue")

                    while True:
                        status = await queue_manager.get_job_status(job_id)

                        if status == "complete":
                            result = await queue_manager.get_job_result(job_id)
                            logger.info(f"Vision job {job_id} completed")

                            content_chunk = format_vision_stream_chunk(
                                result["choices"][0]["message"]["content"],
                                None,
                                request_model,
                                include_role=True,
                            )
                            yield f"data: {json.dumps(content_chunk)}\n\n"

                            finish_chunk = format_vision_stream_chunk(
                                None, "stop", request_model
                            )
                            yield f"data: {json.dumps(finish_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        elif status in ["error", "cancelled", "timeout"]:
                            logger.error(
                                f"Vision job {job_id} ended with status {status}"
                            )
                            yield f"data: {json.dumps({'error': f'Job {status}'})}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in vision stream: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                vision_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            # Non-streaming vision request
            try:
                await queue_manager.add_job(job_id, request_model)
                while True:
                    status = await queue_manager.get_job_status(job_id)
                    if status == "complete":
                        result = await queue_manager.get_job_result(job_id)
                        return JSONResponse(content=result)
                    elif status in ["error", "cancelled", "timeout"]:
                        return JSONResponse(
                            status_code=500, content={"error": f"Job {status}"}
                        )
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing vision request: {str(e)}")
                return JSONResponse(status_code=500, content={"error": str(e)})

    # Handle text models with vLLM if enabled
    elif use_vllm:
        from vllm.sampling_params import SamplingParams

        engine = request.app.state.engine
        sampling_params = SamplingParams(
            temperature=request_model.temperature,
            top_p=request_model.top_p,
            max_tokens=request_model.max_tokens,
            stop=request_model.stop if hasattr(request_model, "stop") else None,
        )

        prompt = format_messages_prompt(request_model.messages)

        if request_model.stream:

            async def vllm_stream():
                try:
                    async for output in engine.generate(
                        prompt, sampling_params, request_id=str(uuid.uuid4())
                    ):
                        if output.outputs[0].text:
                            chunk = format_vllm_stream_chunk(
                                output.outputs[0].text,
                                output.outputs[0].finish_reason,
                                request_model,
                            )
                            yield f"data: {json.dumps(chunk)}\n\n"

                        if output.outputs[0].finish_reason:
                            chunk = format_vllm_stream_chunk(
                                None, output.outputs[0].finish_reason, request_model
                            )
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                except Exception as e:
                    logger.error(f"Error in vLLM stream: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                vllm_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            try:
                generated_text = []
                finish_reason = None

                async for output in engine.generate(
                    prompt, sampling_params, request_id=str(uuid.uuid4())
                ):
                    generated_text.append(output.outputs[0].text)
                    finish_reason = output.outputs[0].finish_reason

                final_text = "".join(generated_text)

                class MockOutput:
                    def __init__(self, text, finish_reason):
                        self.text = text
                        self.finish_reason = finish_reason

                class MockOutputs:
                    def __init__(self, outputs):
                        self.outputs = outputs

                mock_output = MockOutputs([MockOutput(final_text, finish_reason)])
                response = format_vllm_response(mock_output, request_model)
                return JSONResponse(content=response)
            except Exception as e:
                logger.error(f"Error in vLLM generation: {str(e)}")
                return JSONResponse(status_code=500, content={"error": str(e)})

    # Fall back to queue-based processing for text models without vLLM
    else:
        app = request.app
        queue_manager = app.state.queue_manager
        job_id = str(uuid.uuid4())

        if request_model.stream:

            async def queue_stream():
                try:
                    await queue_manager.add_job(job_id, request_model)

                    while True:
                        status = await queue_manager.get_job_status(job_id)

                        if status == "complete":
                            chunks = await queue_manager.get_job_result(job_id)
                            if isinstance(chunks, list):
                                for chunk in chunks:
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        elif status in ["error", "cancelled", "timeout"]:
                            yield f"data: {json.dumps({'error': f'Job {status}'})}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in queue stream: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                queue_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            try:
                await queue_manager.add_job(job_id, request_model)
                while True:
                    status = await queue_manager.get_job_status(job_id)
                    if status == "complete":
                        result = await queue_manager.get_job_result(job_id)
                        return JSONResponse(content=result)
                    elif status in ["error", "cancelled", "timeout"]:
                        return JSONResponse(
                            status_code=500, content={"error": f"Job {status}"}
                        )
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                return JSONResponse(status_code=500, content={"error": str(e)})
