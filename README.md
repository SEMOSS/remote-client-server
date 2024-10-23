# Remote Client Server

This FastAPI application provides a scalable and efficient solution for serving generative AI models requiring GPU acceleration. It supports HTTP clients and can be deployed locally or within a Docker container.

A built-in queuing system manages GPU resources, ensuring requests are processed sequentially in the order received. Horizontal scaling is achieved by deploying multiple instances of the server, allowing you to handle increased demand.

Currently, the server is configured to run one model type at a time. For instructions on adding new models, please refer to the "Adding New Models" section.

## Current Supported Models
The following models are currently supported. Use the `MODEL` environment variable to specify which model to load by including the value of the key value pair of the supported models below.

- Image Generation 
    - MODEL: `PixArt-alpha/PixArt-XL-2-1024-MS` -- SHORTNAME : `pixart`
    - MODEL: `microsoft/Phi-3-mini-128k-instruct` -- SHORTNAME : `phi-3-mini-128k-instruct`
    - MODEL: `urchade/gliner_multi-v2.1` -- SHORTNAME : `gliner-multi-v2.1`

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Downloading Model Files
- The model files are downloaded at the start up of the Docker container it not currently present.
- Additionally, the model files are too large to store more than one model at a time in a Docker container. During deployments, we utilize Docker volumes. Refer to `download.py` for logic on how the model files are checked and downloaded on server start up.
- When developing locally, you can download the model files using the `utils/dl_model_files.py` script or just have the start up lifecycle do it for you (This will remove any existing files in model_files).

## Running the Server Locally
- You can run the server locally using the `server/main.py` script. Make sure to specify the model you want to run using the `--model` flag and the `--local_files` flag to use the local model files.
```bash
python server/main.py --model pixart --local_files
```
- You can specify the host and port using the `--host` and `--port` flags.
```bash
python main.py --model pixart --host "127.0.0.1" --port 5000 --local_files
``` 

## Port 
- Server runs on port `localhost:8888` unless otherwise specified.

## Docker
- You can build the Docker container using the following command. The `INSTALL_FLASH_ATTENTION` argument is used to install the Flash Attention library. If you are not using the Flash Attention library, set this argument to `false` (It takes a really long time to build and is not required).
```bash
docker build --build-arg INSTALL_FLASH_ATTENTION=false -t remote-client-server .
```
If you run the container without a volume attached, make sure the model files are downloaded in the `model_files` directory on build.
```bash
docker run -p 8888:8888 -e MODEL=pixart -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server remote-client-server
```

To run the container with a volume attached with the model files, first create the volume and then use the following command to start the container with the volume attached. 
```bash
docker run --rm -p 8888:8888 -e MODEL=pixart -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server -v pixart-volume:/app/model_files remote-client-server
```

## Docker Volumes
- You can use Docker volumes to store the model files.
- The volume will contain root directories for each model which will be named by the model short name.
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
- Add the a Pydantic model to the `server/pydantic_models` directory to support the new model.
- Update the verify_payload method in `model_utils/model_config.py` to pull the correct Pydantic model on request.
- Update the lifespan event in `server/main.py` to add the model type to the if block and load your new class.


## Formatting
- This project uses the [Black](https://black.readthedocs.io/en/stable/) code formatter. Please install the Black formatter in your IDE to ensure consistent code formatting before submitting a PR.

## TO DO:

- [ ] Update ImageGen class to use generic pipeline and abstract class for different image generation models.
- [ ] Add semaphore and Docker env for setting the number of conncurrent operations utilzing GPU (currently set to 1).

