import os
import asyncio
from fastapi import WebSocket
from typing import Dict, Any
from gaas.image_gen import ImageGen
import logging
import concurrent.futures
from starlette.websockets import WebSocketDisconnect, WebSocketState

logger = logging.getLogger(__name__)


class QueueManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_job = None
        self.websocket_map: Dict[str, WebSocket] = {}
        self.job_status: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def add_job(self, job_id: str, websocket: WebSocket, request: Dict[str, Any]):
        await self.queue.put((job_id, websocket, request))
        self.websocket_map[job_id] = websocket
        self.job_status[job_id] = "queued"
        await self.update_queue_positions()

    async def process_jobs(self):
        while True:
            try:
                self.current_job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            job_id, websocket, request = self.current_job

            try:
                await self.update_queue_positions()

                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info(f"Client disconnected for job {job_id}. Skipping.")
                    continue

                async with self.lock:
                    self.job_status[job_id] = "processing"
                logger.info(f"Processing job {job_id} for request: {request}")

                try:
                    await websocket.send_json(
                        {"status": "processing", "message": "Generating image..."}
                    )

                    # Running image generation in a separate thread
                    loop = asyncio.get_running_loop()
                    response, chunks = await asyncio.wait_for(
                        loop.run_in_executor(self.executor, self.model_switch, request),
                        timeout=90,
                    )

                    await websocket.send_json(response)
                    for i, chunk in enumerate(chunks):
                        await websocket.send_json(
                            {
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "data": chunk,
                            }
                        )

                    await websocket.send_json({"status": "complete"})
                    async with self.lock:
                        self.job_status[job_id] = "complete"
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for job {job_id}")
                    async with self.lock:
                        self.job_status[job_id] = "disconnected"
                except asyncio.TimeoutError:
                    logger.error(f"Job {job_id} timed out")
                    async with self.lock:
                        self.job_status[job_id] = "timeout"
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"error": str(e)})
                    async with self.lock:
                        self.job_status[job_id] = "error"

            finally:
                self.current_job = None
                if job_id in self.websocket_map:
                    del self.websocket_map[job_id]
                self.queue.task_done()
                await self.update_queue_positions()

    async def remove_job(self, job_id: str):
        async with self.lock:
            if job_id in self.job_status:
                del self.job_status[job_id]
            if job_id in self.websocket_map:
                del self.websocket_map[job_id]

        # Remove the job from the queue
        self.queue._queue = [item for item in self.queue._queue if item[0] != job_id]

    async def update_queue_positions(self):
        for i, (job_id, _, _) in enumerate(self.queue._queue):
            websocket = self.websocket_map.get(job_id)
            if websocket:
                message = self.get_queue_size_message(i)
                await websocket.send_json(message)

    def get_queue_size_message(self, queue_position):
        return {
            "status": "waiting",
            "message": f"Your position in the queue is: {queue_position + 1}",
        }

    def get_queue_size(self):
        return self.queue.qsize()

    async def get_job_status(self, job_id: str) -> str:
        async with self.lock:
            return self.job_status.get(job_id, "unknown")

    async def cancel_job(self, job_id: str):
        async with self.lock:
            if job_id in self.job_status:
                self.job_status[job_id] = "cancelled"
            if job_id in self.websocket_map:
                del self.websocket_map[job_id]

    def model_switch(self, request):
        MODEL = os.getenv("MODEL", "image")

        if MODEL.lower() == "image":
            image_gen = ImageGen()
            return image_gen.generate_image(**request)


queue_manager = QueueManager()
