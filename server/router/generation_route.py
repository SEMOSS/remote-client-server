import json
import logging
import uuid
import asyncio
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from sockets.connection_manager import ConnectionManager
from sockets.queue_manager import queue_manager
from pydantic_models.models import ImageRequest
from model_utils.model_config import verify_payload


logger = logging.getLogger(__name__)
manager = ConnectionManager()
generation_router = APIRouter()


@generation_router.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    job_id = None
    try:
        await websocket.send_json({"status": "connected"})
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received data: {data}")
            try:
                request_dict = json.loads(data)
                # Verifying the payload by pulling the model name from the request and pulling the correct pydantic model
                request = verify_payload(request_dict)

                job_id = str(uuid.uuid4())
                await queue_manager.add_job(job_id, websocket, request.dict())

                while True:
                    status = await queue_manager.get_job_status(job_id)
                    if status == "complete":
                        break
                    elif status == "error":
                        await websocket.send_json({"error": "Job processing failed"})
                        break
                    await asyncio.sleep(1)  # Poll every second

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client disconnected")
        if job_id:
            await queue_manager.cancel_job(job_id)
    except Exception as e:
        await manager.disconnect(websocket)
        logger.error(f"Error in WebSocket connection: {e}")
        if job_id:
            await queue_manager.cancel_job(job_id)
