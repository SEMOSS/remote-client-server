from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
