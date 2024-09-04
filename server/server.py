import asyncio
import json
import websockets
from gaas.image_gen import ImageGen
from concurrent.futures import ThreadPoolExecutor
import argparse

request_queue = asyncio.Queue()
pending_connections = asyncio.Queue()
queue_positions = {}
thread_pool = ThreadPoolExecutor(max_workers=4)


async def process_queue():
    while True:
        websocket, request = await request_queue.get()
        try:
            keep_alive_task = asyncio.create_task(send_keep_alive(websocket))

            prompt = request.get("prompt", "A default prompt")
            consistency_decoder = request.get("consistency_decoder", False)
            negative_prompt = request.get("negative_prompt", None)
            guidance_scale = request.get("guidance_scale", 7.5)
            num_inference_steps = request.get("num_inference_steps", 50)
            height = request.get("height", 512)
            width = request.get("width", 512)
            seed = request.get("seed", None)
            file_name = request.get("file_name", "client_x.jpg")

            print("Processing request with prompt:", prompt)

            image_gen = ImageGen()
            try:
                response, chunks = await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    image_gen.generate_image,
                    prompt,
                    consistency_decoder,
                    negative_prompt,
                    guidance_scale,
                    num_inference_steps,
                    height,
                    width,
                    seed,
                    file_name,
                )
            except Exception as e:
                response = {"error": str(e)}
                print("Error processing request:", e)

            response_data = json.dumps(response)

            await websocket.send(response_data)
            for i, chunk in enumerate(chunks):
                await websocket.send(json.dumps({"chunk_index": i, "data": chunk}))

            await websocket.send(json.dumps({"status": "complete"}))

            await asyncio.sleep(1)
            keep_alive_task.cancel()

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))
        finally:
            request_queue.task_done()


async def send_keep_alive(websocket):
    try:
        while True:
            await asyncio.sleep(5)
            await websocket.send(json.dumps({"status": "processing"}))
    except asyncio.CancelledError:
        print("Task finshed properly.")
        pass


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


async def update_queue_positions():
    while True:
        for position, (ws, _) in enumerate(request_queue._queue, start=1):
            try:
                await ws.send(json.dumps({"queue_position": position}))
            except websockets.exceptions.ConnectionClosed:
                pass
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
