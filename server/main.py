import logging
import asyncio
import argparse
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from gaas.model_manager.model_manager import ModelManager
from queue_manager.queue_manager import QueueManager
from gaas.image_gen.image_gen import ImageGen
from gaas.text_gen.chat import Chat
from gaas.ner_gen.ner_gen import NERGen
from gaas.embed_gen.embed_gen import EmbedGen
from gaas.embed_gen.vision_embed_gen import VisionEmbedGen
from gaas.vision_gen.vision_gen import VisionGen
from gaas.sentiment_gen.sentiment_gen import SentimentGen
from globals import app_instance
from router.health_check_route import health_check_router
from router.generation_route import generation_router
from router.queue_route import queue_router
from router.metrics_route import metrics_router
from router.status_route import status_route
from router.embeddings_route import embeddings_router
from router.model_load_check_route import model_loaded_check_router
from router.gpu_status import gpu_status_router
from router.sentiment_gen_route import sentiment_generation_router


# from router.reclaim_route import reclaim_route
from router.chat_completion_route import chat_completion_router
from model_utils.download import (
    check_and_download_model_files,
)
from model_utils.model_config import get_model_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # These are start up events... We can add shutdown events below the yield
    download_success = await asyncio.to_thread(check_and_download_model_files)

    if not download_success:
        logger.error("Failed to download model files")
        # sys.exit(1)
        yield
        return

    # A singleton class representing the single model loaded into memory
    model_manager = ModelManager.get_instance()
    initialized = model_manager.initialize_model()
    if not initialized:
        logger.error("Failed to initialize model")
        # sys.exit(1)
        yield
        return
    else:
        logger.info("Model initialized successfully")

    model_config = get_model_config()
    model_type = model_config.get("type")
    repo_id = model_config.get("model_repo_id")

    if model_type == "image":
        app.state.gaas = ImageGen(model_name=repo_id)
    elif model_type == "text":
        app.state.gaas = Chat(model_manager=model_manager)
    elif model_type == "ner":
        app.state.gaas = NERGen(model_manager=model_manager)
    elif model_type == "embed":
        app.state.gaas = EmbedGen(model_manager=model_manager)
    elif model_type == "vision-embed":
        app.state.gaas = VisionEmbedGen(model_manager=model_manager)
    elif model_type == "vision":
        app.state.gaas = VisionGen(model_manager=model_manager)
    elif model_type == "sentiment":
        app.state.gaas = SentimentGen(model_manager=model_manager)
    else:
        logger.error(f"Unsupported model type: {model_type}")

    app.state.queue_manager = QueueManager(gaas=app.state.gaas)
    app.state.queue_manager_task = asyncio.create_task(
        app.state.queue_manager.process_jobs()
    )
    yield


app = FastAPI(lifespan=lifespan)
app.state.server_status = "initializing server"
app_instance.set_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


app.include_router(health_check_router, prefix="/api")
app.include_router(generation_router, prefix="/api")
app.include_router(status_route, prefix="/api")
app.include_router(queue_router, prefix="/api")
# app.include_router(reclaim_route, prefix="/api")
app.include_router(chat_completion_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(model_loaded_check_router, prefix="/api")
app.include_router(gpu_status_router, prefix="/api")
app.include_router(sentiment_generation_router, prefix="/api")

app.include_router(metrics_router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host IP address")
    parser.add_argument("--port", default=8888, type=int, help="Port number")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--model_repo_id", type=str, help="Hugging Face model id")
    parser.add_argument("--model_type", type=str, help="Model type")
    parser.add_argument(
        "--model_device", type=str, help="Model Device (ex: cpu, cuda (for GPU))"
    )
    parser.add_argument("--semoss_id", type=str, help="Semoss model id")
    parser.add_argument("--no_redis", action="store_true", help="Disable Redis")
    parser.add_argument(
        "--local_files", action="store_true", help="Use local model files"
    )

    args = parser.parse_args()

    if args.model:
        os.environ["MODEL"] = args.model
    if args.model_repo_id:
        os.environ["MODEL_REPO_ID"] = args.model_repo_id
    if args.model_type:
        os.environ["MODEL_TYPE"] = args.model_type
    if args.model_device:
        os.environ["MODEL_DEVICE"] = args.model_device
    if args.semoss_id:
        os.environ["SEMOSS_ID"] = args.semoss_id

    if args.local_files:
        os.environ["LOCAL_FILES"] = "True"
    else:
        os.environ["LOCAL_FILES"] = "False"

    if args.no_redis:
        os.environ["NO_REDIS"] = "True"
    else:
        os.environ["NO_REDIS"] = "False"

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
