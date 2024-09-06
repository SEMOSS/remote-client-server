import os
import asyncio
import json
import websockets
import argparse


request_queue = asyncio.Queue()
TYPE = os.environ.get("TYPE", "image")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE_SIZE_FILE = os.path.join(PROJECT_ROOT, "tmp", "queue_size.txt")


async def process_queue():
    from tcp_server.handlers.image_server_handler import image_server_handler

    if TYPE == "image":
        await image_server_handler(request_queue)
    else:
        print("Invalid type specified.")


def get_current_queue_size():
    return request_queue.qsize()


async def handle_client(websocket, path):
    print(f"Client connected: {path}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")

            try:
                request = json.loads(message)
                await request_queue.put((websocket, request))
            except json.JSONDecodeError:
                print("Failed to decode JSON")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")


def update_queue_size_file():
    with open(QUEUE_SIZE_FILE, "w") as f:
        f.write(str(request_queue.qsize()))


async def update_queue_positions():
    while True:
        for position, (ws, _) in enumerate(request_queue._queue, start=1):
            try:
                await ws.send(json.dumps({"queue_position": position}))
            except websockets.exceptions.ConnectionClosed:
                pass
        update_queue_size_file()
        await asyncio.sleep(1)


async def main(host="0.0.0.0", port=8888):
    asyncio.create_task(process_queue())
    asyncio.create_task(update_queue_positions())

    server = await websockets.serve(
        handle_client,
        host,
        port,
        ping_interval=20,
        ping_timeout=60,
        close_timeout=300,
        max_size=100 * 1024 * 1024,
    )
    print("WebSocket server started on ws://0.0.0.0:8888")
    await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asyncio Python TCP Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="Port to run the server on"
    )
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))
