from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from http_server.router.health_check import health_check_router
from http_server.router.queue_size import queue_size_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_check_router, prefix="/api")
app.include_router(queue_size_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
