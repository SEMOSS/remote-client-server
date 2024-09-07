# Remote Client Server

This is a FastAPI application that supports TCP and HTTP clients. The server can be run locally or in a Docker container. This server is used to serve different gen-AI models that require a GPU for inference. 

This server uses a queue to manage GPU consumption. The queue accepts the websocket connection and processes the request in the order it was received. The server can be scaled horizontally by running multiple instances of the server.

This is currently setup to only run a single type of model at a time. Please see the Adding New Models section for more information on how to add new models.

## Current Supported Models
- ImageGen - PixArt-alpha/PixArt-XL-2-1024-MS

## Local Installation (Assumes Windows w/ Anaconda)
Running PyTorch with CUDA on Windows can be a bit tricky and the steps may vary based on your system configuration. The following steps should help you get started.

- conda activate base
- conda create --name your_environment_name python=3.11
- conda activate your_environment_name
- conda install cuda --channel nvidia/label/cuda-12.4.0
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
- conda env update -f environment.yml
- pip install -r requirements.txt

## Downloading Model Files
- The model files for PixArt-alpha/PixArt-XL-2-1024-MS can be downloaded locally using the `utils/dl_model_files.py` script. The script will download the model files to the `models_files` directory.
- This prevents the Docker container from having to download the model files each time it starts up.
- I can't push the model files to GitHub because they are too large so you will need to download them locally before building the Docker container.

## Running the Server Locally
- You can run the server locally using the `server/main.py` script.
```bash
python server/main.py
```
- You can specify the host and port using the `--host` and `--port` flags.
```bash
python main.py --host "127.0.0.1" --port 5000 
``` 

## Port 
- Server runs on port `localhost:8888` unless otherwise specified.

## Docker
```bash
docker build -t remote-client-server .
```

```bash
docker run -p 8888:8888 -e TYPE=image -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server remote-client-server
```

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Access API Documentation
- `http://127.0.0.1:8888/docs` for Swagger UI documentation.
- `http://127.0.0.1:8888/redoc` for ReDoc documentation.

## Access API Endpoints
- `ws://localhost:8888/api/generate` - Gen-AI WebSocket endpoint.

- `http://localhost:8888/api/gpu` - GPU status endpoint (Returns whether the PyTorch can access GPU on the container).

- `http://localhost:8888/api/health` - Health check endpoint.

- `http://localhost:8888/api/queue_size` - Queue size endpoint (Returns size of the queue).


## Adding New Models
- Add a new file and class to the `app/gaas` directory to support the new model.
- Update the `model_switch()` method in the QueueManager class to support the new model.
- You do NOT need to add an additional endpoint.
- You can enforce type checking with pydantic by adding a new class to the `server/models` directory.


## Formatting
- This project uses the [Black](https://black.readthedocs.io/en/stable/) code formatter. Please install the Black formatter in your IDE to ensure consistent code formatting before submitting a PR.

## TO DO:

- [ ] Update ImageGen class to use generic pipeline and abstract class for different image generation models.
- [ ] Update the generation route for dynamically type checking the request for different models.
- [ ] Add semaphore and Docker env for setting the number of conncurrent operations utilzing GPU (currently set to 1).
- [ ] Look into multi-stage Docker builds for reducing image size.

