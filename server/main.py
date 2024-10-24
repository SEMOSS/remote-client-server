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
from queue_manager.queue_manager import QueueManager
from gaas.image_gen.image_gen import ImageGen
from gaas.text_gen.text_gen import TextGen
from gaas.gliner_gen.gliner_gen import GlinerGen
from globals import app_instance
from router.health_check_route import health_check_router
from router.generation_route import generation_router
from router.queue_route import queue_router
from router.metrics_route import metrics_router
from router.status_route import status_route
from router.reclaim_route import reclaim_route
from model_utils.download import check_and_download_model_files
from model_utils.model_config import get_model_type, get_repo_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # These are start up events... We can add shutdown events below the yield
    await asyncio.to_thread(check_and_download_model_files)
    # Instantiate the GAAS generation model into memory based on the startup model type
    model_type = get_model_type()
    repo_id = get_repo_id()
    if model_type == "image":
        app.state.gaas = ImageGen(model_name=repo_id)
    elif model_type == "text":
        app.state.gaas = TextGen(model_name=repo_id)
    elif model_type == "NER":
        app.state.gaas = GlinerGen(model_name=repo_id)

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
app.include_router(reclaim_route, prefix="/api")
app.include_router(metrics_router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host IP address")
    parser.add_argument("--port", default=8888, type=int, help="Port number")
    parser.add_argument("--model", default="pixart", type=str, help="Model name")
    parser.add_argument(
        "--local_files", action="store_true", help="Use local model files"
    )

    args = parser.parse_args()

    if args.model:
        os.environ["MODEL"] = args.model

    if args.local_files:
        os.environ["LOCAL_FILES"] = "True"
    else:
        os.environ["LOCAL_FILES"] = "False"

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
