import os
import logging
import gc
import asyncio
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from gaas.image_gen.image_gen import ImageGen
from gaas.ner_gen.ner_gen import NERGen
from gaas.text_gen.text_gen_factory import TextGenFactory
from queue_manager.queue_manager import QueueManager
from model_utils.model_config import SUPPORTED_MODELS, get_model_type, get_repo_id
from model_utils.download import check_and_download_model_files


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reclaim_route = APIRouter()


@reclaim_route.post("/reclaim")
async def reclaim_model(request: Request):
    data = await request.json()
    new_model_name = data.get("model_name")
    if not new_model_name:
        return JSONResponse(
            status_code=400, content={"error": "Model name not provided"}
        )

    app = request.app

    if new_model_name not in SUPPORTED_MODELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported model name '{new_model_name}'"},
        )

    try:
        app.state.queue_manager_task.cancel()
        try:
            await app.state.queue_manager_task
        except asyncio.CancelledError:
            pass

        del app.state.gaas
        # Free GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        os.environ["MODEL"] = new_model_name

        await asyncio.to_thread(check_and_download_model_files)

        model_type = get_model_type()
        repo_id = get_repo_id()
        if model_type == "image":
            app.state.gaas = ImageGen(model_name=repo_id)
        elif model_type == "text":
            app.state.gaas = TextGenFactory(model_name=repo_id)
        elif model_type == "ner":
            app.state.gaas = NERGen(model_name=repo_id)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown model type for '{new_model_name}'"},
            )

        app.state.queue_manager = QueueManager(gaas=app.state.gaas)
        app.state.queue_manager_task = asyncio.create_task(
            app.state.queue_manager.process_jobs()
        )

        return {"message": f"Model has been switched to '{new_model_name}'"}

    except Exception as e:
        logging.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
