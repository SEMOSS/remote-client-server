# Remote Client Server

This is a FastAPI application that supports HTTP clients. The server can be run locally or in a Docker container. This server is used to serve different gen-AI models that require a GPU for inference. 

This server uses a queue to manage GPU consumption. The queue accepts the connection and processes the request in the order it was received. The server can be scaled horizontally by running multiple instances of the server.

This is currently setup to only run a single type of model at a time. Please see the Adding New Models section for more information on how to add new models.

## Current Supported Models
The following models are currently supported. Use the `MODEL` environment variable to specify which model to load by including the value of the key value pair of the supported models below.

- Image Generation 
    - MODEL: `PixArt-alpha/PixArt-XL-2-1024-MS` -- SHORTNAME : `pixart`
    - MODEL: `Phi-3-mini-128k-instruct` -- SHORTNAME : `phi-3-mini-128k-instruct`

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Downloading Model Files
- The model files are too large to store on github and are downloaded at the start up of the Docker container.
- Additionally, the model files are too large to store more than one model at a time in a Docker container. Refer to `download.py` for logic on how the model files are checked and downloaded on server start up.
- When developing locally, you can download tdhe model files using the `utils/dl_model_files.py` script or just have the start up lifecycle do it for you (This will remove any existing files in model_files).

## Running the Server Locally
- You can run the server locally using the `server/main.py` script.
```bash
python server/main.py --model pixart --local_files
```
- Specify the model
```
python server/main.py --model "phi-3-mini-128k-instruct"
```
- You can specify the host and port using the `--host` and `--port` flags.
```bash
python main.py --host "127.0.0.1" --port 5000 --local_files
``` 

## Port 
- Server runs on port `localhost:8888` unless otherwise specified.

## Docker
```bash
docker build -t remote-client-server .
```
If you run the container without a volume attached, make sure the model files are downloaded in the `model_files` directory.
```bash
docker run -p 8888:8888 -e MODEL=pixart -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server remote-client-server
```

Run the container with a volume attached with the model files. 
```bash
docker run --rm -p 8888:8888 -e MODEL=pixart -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server -v pixart-volume:/app/model_files remote-client-server
```

## Docker Volumes
- You can use Docker volumes to store the model files.
- The volume should contain root directories for each model which should be named by the model short name.
- The volume is attached to the container at `/app/model_files`.

## Access API Documentation
- `http://127.0.0.1:8888/docs` for Swagger UI documentation.
- `http://127.0.0.1:8888/redoc` for ReDoc documentation.

## Access API Endpoints
- `http://localhost:8888/api/generate` - Gen-AI generation endpoint.

- `http://localhost:8888/api/health` - Health check endpoint.

- `http://localhost:8888/api/status` - Returns an object with values for the current model, queue size, GPU utilization and server status.

- `http://localhost:8888/api/queue` - Returns the current queue size as a plain text response. (IE: "queue_size 0")

- `http://localhost:8888/metrics` - Returns Prometheus metrics.


## Adding New Models
- Add a new file and class to the `app/gaas` directory to support the new model.
- In `model_utils/model_config.py`, add the model config to the `SUPPORTED_MODELS` object.
- The expected_model_class can be found in the `model_index.json` of the downloaded model files.
- If you are adding a new `TYPE` of model, update the `model_switch()` method in the QueueManager class to support the new model.
- You can enforce type checking with pydantic by adding a new class to the `server/pydantic_models` directory.


## Formatting
- This project uses the [Black](https://black.readthedocs.io/en/stable/) code formatter. Please install the Black formatter in your IDE to ensure consistent code formatting before submitting a PR.

## TO DO:

- [ ] Update ImageGen class to use generic pipeline and abstract class for different image generation models.
- [ ] Add semaphore and Docker env for setting the number of conncurrent operations utilzing GPU (currently set to 1).

