import asyncio
import json
import websockets

CLIENT_NUMBER = 1


async def websocket_client():
    uri = "ws://localhost:8888"
    async with websockets.connect(uri) as websocket:
        request = {
            "prompt": "A scenic mountain landscape.",
            "file_name": f"client{CLIENT_NUMBER}",
        }
        request_data = json.dumps(request)

        print(f"Sending: {request_data}")
        await websocket.send(request_data)

        chunks = []
        total_chunks = None
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=300)

                response_data = json.loads(response)

                if "error" in response_data:
                    raise Exception(
                        f"Server error on Client {CLIENT_NUMBER}: {response_data['error']}"
                    )

                if "total_chunks" in response_data:
                    total_chunks = response_data["total_chunks"]

                if "chunk_index" in response_data:
                    chunks.append(response_data["data"])

                    if len(chunks) == total_chunks:
                        print("All chunks received. Reconstructing image...")
                        base64_uncompressed = "".join(chunks)
                        break

                if "total_chunks" in response_data:
                    # I don't want to print the base64 string because it's too large
                    response_obj = {
                        "CLIENT": {CLIENT_NUMBER},
                        "generation_time": response_data.get("generation_time"),
                        "seed": response_data.get("seed"),
                        "prompt": response_data.get("prompt"),
                        "negative_prompt": response_data.get("negative_prompt"),
                        "guidance_scale": response_data.get("guidance_scale"),
                        "num_inference_steps": response_data.get("num_inference_steps"),
                        "height": response_data.get("height"),
                        "width": response_data.get("width"),
                        "model_name": response_data.get("model_name"),
                        "vae_model_name": response_data.get("vae_model_name"),
                    }
                    print("Response: ", response_obj)

                elif response_data.get("status") == "processing":
                    print(f"Still processing Client {CLIENT_NUMBER}...")

            except asyncio.TimeoutError:
                print(f"Timeout waiting for server response on Client {CLIENT_NUMBER}.")
                break

            except websockets.exceptions.ConnectionClosed as e:
                print(f"Client {CLIENT_NUMBER} connection closed unexpectedly: {e}")
                break


asyncio.run(websocket_client())
