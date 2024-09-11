import logging
import asyncio
import argparse
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sockets.queue_manager import queue_manager
from router.health_check_route import health_check_router
from router.generation_route import generation_router
from router.status_route import status_route
from router.models_route import models_route
from model_utils.download import verify_model_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # These are start up events... We can add shutdown events below the yield
    await asyncio.to_thread(verify_model_files, "pixart", True)
    asyncio.create_task(queue_manager.process_jobs())
    yield


app = FastAPI(lifespan=lifespan)

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
app.include_router(models_route, prefix="/api")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host IP address")
    parser.add_argument("--port", default=8888, type=int, help="Port number")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
