# Remote Client Server

This FastAPI server provides a scalable and efficient solution for serving generative AI models requiring GPU acceleration. It supports HTTP clients and can be deployed locally or within a Docker container. 

The service implements OpenAI-compatible endpoints, allowing seamless integration with existing tools and libraries.

A built-in queuing system manages GPU resources, ensuring requests are processed sequentially in the order received. Horizontal scaling is achieved by deploying multiple instances of the server, allowing you to handle increased demand.

The server is configured to run one model type at a time. The model type is specified at startup and determines the model's behavior and the type of requests it can handle. The server supports multiple model types, including LLMs, NER, text and image embeddings, and vision models.

## Currently Supported Model Types
- **LLMs** 
    - KEY: `text`
    - IE: *microsoft/Phi-3.5-mini-instruct*
- **NER** (Entity Recognition) 
    - KEY: `ner`
    - IE: *urchade/gliner_medium-v2.1*
- **Text Embedding** 
    - KEY: `embed`
    - IE: *nomic-ai/nomic-embed-text-v1.5*
- **Image Embedding** 
    - KEY: `vision-embed`
    - IE: *nomic-ai/nomic-embed-vision-v1.5*
- **Vision**
    - KEY: `vision`
    - IE: *microsoft/Florence-2-large*
- **Image**
    - KEY: `image`
    - IE: *PixArt-alpha/PixArt-XL-2-1024-MS*

## Formatting
- This project uses the [Black](https://black.readthedocs.io/en/stable/) code formatter. Please install the Black formatter in your IDE to ensure consistent code formatting before submitting a PR.

## Python Install
You can install the required Python packages using UV or a standard Python virtual environment. The project requires Python 3.10 or higher. When adding new Python packages, please update the `dev_requirements.txt` file and the `pyproject.toml` file.
- Local Python install on Windows:
```bash
python -m venv venv

.\venv\Scripts\activate

pip install -r dev_requirements.txt
```
- Local Python install on MacOS/Linux:
```bash
python3 -m venv venv

source venv/bin/activate

pip install -r dev_requirements.txt
```

- You can also use the `uv` tool to create a virtual environment and install the required packages.
```bash
uv venv

.venv\Scripts\activate

uv pip install -r pyproject.toml
```

## PyTorch/CUDA
- You can test your local PyTorch/CUDA installation by using the `utils/torch_test.ipynb` notebook.

## Downloading Model Files
- The model files for a specific model are downloaded on the start up of the server if it not currently present. The model files location is determined by the start up configuration and are either placed in the `model_files` directory at the root of the project or in a Docker volume location.
- When developing locally, if you prefer, you can manually download the model files using the `utils/dl_model_files.py` script. The start up lifecycle will then check the model_files directory to ensure the files are present.

## Start Commmand Arguments
- **--model**: A simplified, path-safe name for the model (e.g., '*phi-3-mini-128k-instruct*' for '*microsoft/Phi-3.5-mini-instruct*'). Only use alphanumeric characters and hyphens (-). Avoid special characters like /, @, or spaces.
- **--model_repo_id**: The complete repository ID from Hugging Face Hub (e.g., '*microsoft/Phi-3.5-mini-instruct*'). This should match exactly how it appears on Hugging Face, including the namespace and forward slashes.
- **--model_type**: The type of model being served (e.g., 'text', 'ner', 'embed', 'vision-embed', 'vision', 'image'). Must match one of the supported model type keys listed in the "Currently Supported Model Types" section.
- **semoss_id** (*OPTIONAL*): The SEMOSS Engine ID associated to the model. If you are running locally and do not have one you can omitt this argument.
- **--host** (*OPTIONAL*): The host IP address for the server. Defaults to '0.0.0.0'. This can be omitted in most cases.
- **--port** (*OPTIONAL*): The port number for the server. Defaults to '8888'. This can be omitted in most cases.
- **--local_files** (*OPTIONAL*): A flag indicating that model files should be loaded from the project's root directory rather than from a Docker volume location. When this flag is used, the server expects model files to be present in the local model_files directory at the root instead of looking for them in mounted volumes. Useful for local development and testing.
- **--no_redis** (*OPTIONAL RECOMMENDED*): A flag to disable Redis deployment status updates. When running the server locally without Redis infrastructure, use this flag to prevent the server from attempting to connect to and update Redis with deployment status information. This is typically used during local development and testing scenarios where Redis is not available or needed.


## Running the Server Locally
The following are example commands for running the server locally (*not in a Docker container*) and should be executed at the root of the project.
- EX: Running the Nomic Text Embedding model locally with the model files stored in the `model_files` directory.
```bash
python server/main.py --model nomic-ai-nomic-embed-text-v1-5 --model_repo_id nomic-ai/nomic-embed-text-v1.5 --model_type embed --semoss_id 2aa0e4bf-08d5-452e-aa75-dd417f8ae610 --local_files --no_redis
```
- EX: Running the GLiNER NER model locally with the model files stored in the `model_files` directory. Note how we did not include the `--no_redis` flag, so this would assume you have a Redis instance running locally.
```bash
python server/main.py --model gliner-multi-v2-1 --model_repo_id urchade/gliner_multi-v2.1 --model_type ner --semoss_id abd20c47-2ce7-45ef-a10a-572150a3b0d6 --local_files
```

## Docker
The project uses a two-stage Docker build process to optimize build times and reduce the final image size. This process involves creating a base image with common dependencies first, then building the server image on top of it.
### Base Image
The base image (defined in `Dockerfile.base`) contains:

- NVIDIA CUDA runtime
- Python environment setup
- Common ML dependencies (PyTorch, Transformers, etc.)
- Flash Attention configuration
- Git LFS support

To build the base image (*This will take a long time due to Flash Attention installation*):
```bash
docker build -f Dockerfile.base -t remote-client-server-base:latest .
```
### Server Image
The server image (defined in `Dockerfile`) builds upon the base image and adds:

- Application code
- Server configurations
- Model file management
- Additional dependencies

Before building the server image, you need to modify the Dockerfile. There are two FROM lines at the top:
```Dockerfile
# FROM docker.semoss.org/genai/remote-client-server-base:latest
FROM remote-client-server-base
```
- For production builds, use the first line (pointing to the remote registry)
- For local development, comment out the first line and uncomment the second line
- (*IMPORTANT*) When you push changes to the remote repository, remember to switch back to the first line before building the server image.

To build the server image locally:
```bash
docker build -t remote-client-server:latest .
```
### Running with Docker Volumes
To run the container with a volume attached with the model files, first create the volume and then use the following command to start the container with the volume attached. (NOTE: The volume name is `my-volume` in this example, you can name a volume whatever you like).
```bash
docker run -p 8888:8888 -e MODEL=gliner-multi-v2-1 -e MODEL_REPO_ID=urchade/gliner_multi-v2.1 -e MODEL_TYPE=ner -e SEMOSS_ID=abd20c47-2ce7-45ef-a10a-572150a3b0d6 --gpus all --name remote-client-server -v my-volume:/app/model_files remote-client-server
```

## Access API Documentation
- `http://127.0.0.1:8888/docs` for Swagger UI documentation.
- `http://127.0.0.1:8888/redoc` for ReDoc documentation.

## Endpoints
- `/api/chat/completions` - An OpenAI API compatible endpoint for chat completions (text models).

- `/api/embeddings` - An OpenAI API compatible endpoint for text embeddings.

- `/api/generate` - A generic endpoint for generations from models not natively supported by the OpenAI API. IE: NER models.

- `/api/health` - Health check endpoint.

- `/api/status` - Returns an object with values for the current model, queue size, GPU utilization and server status.

- `/api/queue` - Returns the current queue size as a plain text response. (IE: "queue_size 0")

- `/metrics` - Returns Prometheus metrics.

## Redis
The server integrates with Redis to maintain deployment status and facilitate scaling operations. Each model deployment is associated with a Redis hash that stores critical operational metrics and status information.

### Deployment Hash Structure
Each deployment maintains a Redis hash using the SEMOSS Engine ID as the key, with the format {semoss_id}:deployment. The hash contains the following fields:

- `model_name`: The simplified path-safe name of the model
- `model_repo_id`: The complete HuggingFace repository ID
- `model_type`: The type of model being served (text, ner, embed, etc.)
- `semoss_id`: The unique SEMOSS Engine ID associated with the deployment
- `address`: The IP address and port where the model is accessible
- `start_time`: Timestamp when the deployment was initiated
- `last_request`: Timestamp of the most recent generation request
- `generations`: Counter of total generations performed
- `shutdown_lock`: Flag indicating if the deployment is exempt from scaling down

Example Redis hash:
```redis
"abd20c47-2ce7-45ef-a10a-572150a3b0d6:deployment" : {
    "model_name": "gliner-multi-v2-1",
    "model_repo_id": "urchade/gliner_multi-v2.1",
    "model_type": "ner",
    "semoss_id": "abd20c47-2ce7-45ef-a10a-572150a3b0d6",
    "address": "10.218.221.138:31213",
    "start_time": "2025-01-16T17:41:32.934832-05:00",
    "last_request": "2025-01-17T09:43:04.588880-05:00",
    "generations": "588",
    "shutdown_lock": "false"
}
```
### Automatic Updates
The server automatically updates specific fields in the Redis hash during generation operations:

- Generation Counter: Incremented after each successful model generation
- Last Request Time: Updated with the timestamp of the most recent request

These updates enable monitoring systems to track usage patterns and make scaling decisions based on actual deployment activity.

## Project Architecture
This is not comprehensive, but a high-level overview of the project structure.
- `server`: Contains the main server application code.
    - `server/main.py`: The entry point for the server.
    - `server/gaas`: Contains the classes for running specific models (text, ner, embed, etc.) and managing the models.
        - `server/gaas/model_manager/model_manager.py`: The **ModelManager** class is a singleton that manages the model instance and loads the model into memory. 
        - `server/gaas/model_manager/model_files_manager.py`: The **ModelFilesManager** manages the location and parsing of model files.
    - `server/model_utils`
        - `server/model_utils/download.py`: Contains all of the logic for downloading model files and verifying their integrity during the server start up lifecycle.
        - `server/model_utils/model_config.py`: Contains the configuration for the currently running model by pulling the OS environment variables.
    - `server/pydantic_models`: Contains the Pydantic models for requests and response validation.
    - `server/queue_manager`: Contains the **QueueManager** singleton class for managing the job queue.
    - `server/redis_manager`: Contains the **RedisManager** class for managing the Redis connection and deployment status updates.
    - `server/router` Contains the FastAPI routers for the different endpoints.

## Lifecycle Example
The following is a high-level overview of the server lifecycle when starting up and processing requests when the container is started with the following command:
```
docker run -p 8888:8888 -e MODEL=gliner-multi-v2-1 -e MODEL_REPO_ID=urchade/gliner_multi-v2.1 -e MODEL_TYPE=ner -e SEMOSS_ID=abd20c47-2ce7-45ef-a10a-572150a3b0d6 --gpus all --name remote-client-server -v my-volume:/app/model_files remote-client-server
```
### Start Up Lifecycle
1. Model File Verification
    - The server uses the shortname of the model to check if an existing path exists on the volume. (see `download.py`)
        - If the path does not exist, the server will attempt to download the model files from the Hugging Face Hub.
        - If the path exists, the server will verify the integrity of the model files.
2. Model Loading
    - The server will load the model into memory using the **ModelManager** class.
        - The **ModelManager** class will use the OS environment variables to determine the model configuration.
        - The **ModelManager** and **ModelFilesManager** classes will parse the model files to determine whether to utilize Flash Attention.
3. Queue Initialization
    - The server will initialize the **QueueManager** class to manage the job queue based on the model type.

### Request Processing Lifecycle
1. Request Received
    - The server will receive a request at a given endpoint (eg. `/api/embeddings) and validate it using the Pydantic models.
    - The request will be assigned a unique job id and added to the queue.
2. Queue Processing
    - The **QueueManager** will process the job queue sequentially.
    - The **QueueManager** will report the current queue position as it updates.
    - The job will be popped from the queue and sent to the **ModelManager** for processing.
3. Model Processing
    - The **ModelManager** will process the job using the model instance.
    - The **ModelManager** will return the result to the **QueueManager**.
4. Queue Update
    - The **QueueManager** will update the job status with the payload and set the job as complete.
    - The **QueueManager** will update the Redis deployment hash with the latest request timestamp and generation count.
5. Response
    - The server will return the response to the client.

## More Information
- For more information on how the server is deployed, see the [Kubernetes Model Scaler](https://github.com/SEMOSS/kubernetes-model-scaler)

## Contributing
- Please create a new branch for your changes and submit a pull request for review.
    - In the PR description, please include a brief summary of the changes and any relevant information.
    - Ensure that your code is formatted using the Black code formatter.
- PRs will require review from at least one team member (*Ryan Weiler or Kunal Patel for now*).