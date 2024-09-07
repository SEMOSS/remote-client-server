from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Accepting client connection: {websocket}. There are {len(self.active_connections)} active connections."
        )

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(
            f"Disconnecting client connection: {websocket}. There are {len(self.active_connections)} active connections."
        )

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
