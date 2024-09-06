# Docker TCP ImageGen Server

This is a TCP server created with asyncio & websockets to run in a Docker container.

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
- You can run the server locally using the `server/server.py` script.
```bash
python server/server.py
```

## Port 
- Server runs on port `localhost:8888` unless otherwise specified.

## Docker
```bash
docker build -t remote-client-server .
```

```bash
docker run -p 8888:8888 -e HOST=0.0.0.0 -e PORT=8888 --gpus all --name remote-client-server remote-client-server
```

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Testing
- You can test the ImageGen class directly using the `app/test/test_local_image_dl.py` script.

## HTTP Server
- HTTP Server is a FastAPI application that can be run using the `http_server/main.py` script.

### Access API Documentation
- `http://127.0.0.1:8000/docs` for Swagger UI documentation.
- `http://127.0.0.1:8000/redoc` for ReDoc documentation.