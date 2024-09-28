# Remote Client Server

This is a FastAPI application that supports HTTP clients. The server can be run locally or in a Docker container. This server is used to serve different gen-AI models that require a GPU for inference. 

This server uses a queue to manage GPU consumption. The queue accepts the connection and processes the request in the order it was received. The server can be scaled horizontally by running multiple instances of the server.

This is currently setup to only run a single type of model at a time. Please see the Adding New Models section for more information on how to add new models.

## Current Supported Models
The following models are currently supported. Use the `MODEL` environment variable to specify which model to load by including the value of the key value pair of the supported models below.

- Image Generation 
    - MODEL: `PixArt-alpha/PixArt-XL-2-1024-MS` -- SHORTNAME : `pixart`

## Local Installation (Assumes Windows w/ Anaconda)
Running PyTorch with CUDA on Windows can be a bit tricky and the steps may vary based on your system configuration. The following steps should help you get started.

- `conda activate base`
- `conda create --name your_environment_name python=3.11`
- `conda activate your_environment_name`
- `conda install cuda --channel nvidia/label/cuda-12.4.0`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- `conda env update -f environment.yml`
- `pip install -r requirements.txt`

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Downloading Model Files
- The model files are too large to store on github and are downloaded at the start up of the Docker container.
- Additionally, the model files are too large to store more than one model at a time in a Docker container. Refer to `download.py` for logic on how the model files are checked and downloaded on server start up.
- When developing locally, you can download the model files using the `utils/dl_model_files.py` script or just have the start up lifecycle do it for you (This will remove any existing files in model_files).

## Running the Server Locally
- You can run the server locally using the `server/main.py` script.
```bash
python server/main.py MODEL=pixart
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

- `http://localhost:8888/api/models/{model}` - Takes a model short name as a parameter and returns whether the correct model files are present in the model_files directory.


## Adding New Models
- Add a new file and class to the `app/gaas` directory to support the new model.
- In `model_utils/model_config.py`, add the model config to the `SUPPORTED_MODELS` object.
- The expected_model_class can be found in the `model_index.json` of the downloaded model files.
- If you are adding a new `TYPE` of model, update the `model_switch()` method in the QueueManager class to support the new model.
- You can enforce type checking with pydantic by adding a new class to the `server/pydantic_models` directory.


## Formatting
- This project uses the [Black](https://black.readthedocs.io/en/stable/) code formatter. Please install the Black formatter in your IDE to ensure consistent code formatting before submitting a PR.

## Long Path

- If you get an error related to installing the xformers package and the error is related to the Windows long path setting, follow this [tutorial](https://medium.com/@mariem.jabloun/how-to-fix-python-package-installation-long-path-support-os-error-59ab7e9bf10a)
