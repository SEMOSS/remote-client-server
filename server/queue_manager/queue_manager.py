import asyncio
import logging
import concurrent.futures
from typing import Dict, Any
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Define the Gauge metric for tracking queue size
queue_size_gauge = Gauge("modelserver_queue_size", "Current size of the job queue")


class QueueManager:
    def __init__(self, gaas):
        self.gaas = gaas
        self.queue = asyncio.Queue()
        self.current_job = None
        self.job_status: Dict[str, str] = {}
        self.job_results: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def add_job(self, job_id: str, request: Dict[str, Any]):
        await self.queue.put((job_id, request))
        self.job_status[job_id] = "queued"
        queue_size_gauge.set(self.queue.qsize())

    async def process_jobs(self):
        while True:
            try:
                self.current_job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            job_id, request = self.current_job

            try:
                async with self.lock:
                    self.job_status[job_id] = "processing"
                logger.info(f"Processing job {job_id} for request: {request}")

                try:
                    # Running image generation in a separate thread
                    loop = asyncio.get_running_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.executor, self.model_switch, request),
                        timeout=300,
                    )

                    # Storing results for HTTP requests
                    self.job_results[job_id] = result

                    async with self.lock:
                        self.job_status[job_id] = "complete"
                except asyncio.TimeoutError:
                    logger.error(f"Job {job_id} timed out")
                    async with self.lock:
                        self.job_status[job_id] = "timeout"
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}")
                    async with self.lock:
                        self.job_status[job_id] = "error"

            finally:
                self.current_job = None
                self.queue.task_done()
                queue_size_gauge.set(self.queue.qsize())

    async def get_job_position(self, job_id: str) -> int:
        return next(
            (i for i, (jid, _) in enumerate(self.queue._queue) if jid == job_id), -1
        )

    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        return self.job_results.get(job_id, {})

    async def remove_job(self, job_id: str):
        async with self.lock:
            if job_id in self.job_status:
                del self.job_status[job_id]
            if job_id in self.job_results:
                del self.job_results[job_id]

        # Removing job from queue
        self.queue._queue = [item for item in self.queue._queue if item[0] != job_id]
        queue_size_gauge.set(self.queue.qsize())

    def get_queue_size(self):
        return self.queue.qsize()

    async def get_job_status(self, job_id: str) -> str:
        async with self.lock:
            return self.job_status.get(job_id, "unknown")

    async def cancel_job(self, job_id: str):
        async with self.lock:
            if job_id in self.job_status:
                self.job_status[job_id] = "cancelled"
            if job_id in self.job_results:
                del self.job_results[job_id]
        queue_size_gauge.set(self.queue.qsize())

    def model_switch(self, request):
        return self.gaas.generate(request)
