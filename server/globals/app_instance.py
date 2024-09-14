from fastapi import FastAPI

app: FastAPI = None


def set_app(fastapi_app: FastAPI):
    global app
    app = fastapi_app


def get_app() -> FastAPI:
    return app
