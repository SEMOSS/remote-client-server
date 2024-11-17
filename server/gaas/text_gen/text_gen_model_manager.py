import os
import logging
from transformers import AutoModelForCausalLM, pipeline
from globals.globals import set_server_status
from model_utils.model_config import get_short_name
from gaas.tokenizer.tokenizer import Tokenizer
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelManager:
    """
    This is a singleton class that manages the model and tokenizer instances.
    This class is used to ensure that the model and tokenizer are only loaded once.
    """

    _instance = None
    _model = None
    _tokenizer = None
    _pipe = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if ModelManager._instance is not None:
            raise Exception("This class is a singleton!")
        ModelManager._instance = self

    def initialize_model(self, flash_attn_available: bool):
        logger.info("Initializing model!")
        if self._model is None:
            try:
                model_kwargs = {
                    "device_map": "cuda",
                    "torch_dtype": "auto",
                    "trust_remote_code": True,
                }
                if flash_attn_available:
                    model_kwargs["attn_implementation"] = "flash_attention_2"

                short_name = get_short_name()
                model_files_local = os.environ.get("LOCAL_FILES") == "True"

                if model_files_local:
                    # Get the absolute path of the project root directory
                    script_dir = Path(__file__).resolve().parent  # gaas/text_gen
                    project_root = (
                        script_dir.parent.parent.parent
                    )  # Go up to project root
                    model_files_path = project_root / "model_files" / short_name
                    # Convert to string and normalize
                    model_files_path = str(model_files_path.resolve())
                else:
                    model_files_path = f"/app/model_files/{short_name}"

                logger.info(f"Attempting to load model from path: {model_files_path}")

                # Debug checks
                if not os.path.exists(model_files_path):
                    logger.error(
                        f"Model directory does not exist at: {model_files_path}"
                    )
                    raise FileNotFoundError(
                        f"Model directory not found: {model_files_path}"
                    )

                logger.info(
                    f"Model directory exists. Contents: {os.listdir(model_files_path)}"
                )

                # Check for specific required files
                required_files = ["config.json", "generation_config.json"]
                missing_files = [
                    f
                    for f in required_files
                    if not os.path.exists(os.path.join(model_files_path, f))
                ]
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    raise FileNotFoundError(
                        f"Missing required files in model directory: {missing_files}"
                    )

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_files_path,
                    **model_kwargs,
                )
                self._tokenizer = Tokenizer().tokenizer
                self._pipe = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                set_server_status("Model initialization FAILED")
                raise

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def pipe(self):
        return self._pipe
