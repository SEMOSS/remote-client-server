"""
    This module contains variables and methods required for the server handlers.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

thread_pool = ThreadPoolExecutor(max_workers=4)
semaphore = asyncio.Semaphore(1)  # Amount of concurrent requests allowed to utilize GPU


async def send_keep_alive(websocket):
    """
    Keep-alive task to ensure the connection is maintained with the client.
    Args:
        websocket (websockets.WebSocketServerProtocol): The websocket connection
    """
    try:
        while True:
            await asyncio.sleep(5)
            await websocket.send(json.dumps({"status": "processing"}))
    except asyncio.CancelledError:
        print("Keep-alive task finished properly.")
        pass
