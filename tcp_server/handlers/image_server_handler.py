import asyncio
import json
from tcp_server.gaas.image_gen import ImageGen
from tcp_server.handlers.server_utils import thread_pool, semaphore, send_keep_alive


async def image_server_handler(request_queue):

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
                async with semaphore:
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
