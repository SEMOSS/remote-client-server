import os
import logging
import torch
from transformers import AutoModelForCausalLM, pipeline
from gliner import GLiNER
from pathlib import Path
from model_utils.model_config import get_flash_attention, get_model_config
from gaas.tokenizer.tokenizer import Tokenizer
from globals.globals import set_server_status

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class that manages model, tokenizer, and device settings.
    Centralizes all model-related initialization and configuration.
    """

    _instance = None
    _initialized = False
    _model = None
    _tokenizer = None
    _pipe = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            # Initialize device once during instance creation
            cls._instance._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _check_flash_attention(self):
        """Check if flash attention is available and should be used."""
        use_flash_attention = get_flash_attention()
        if not use_flash_attention:
            logger.info("Flash attention is not used for this model.")
            return False
        try:
            import flash_attn  # type: ignore

            logger.info("Flash attention is available.")
            return True
        except ImportError:
            logger.warning("Flash attention is not available.")
            return False

    def initialize_model(self):
        """Initialize the model, tokenizer, and pipeline if not already initialized."""
        if self._initialized:
            logger.debug("Model already initialized, skipping initialization")
            return

        logger.info("Initializing model!")
        try:
            # Checking whether flash attention is available on the container
            flash_attn_available = self._check_flash_attention()
            model_kwargs = {
                "device_map": "cuda",
                "torch_dtype": "auto",
                "trust_remote_code": True,
            }
            if flash_attn_available:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model_config = get_model_config()
            short_name = model_config.get("short_name")

            model_files_local = os.environ.get("LOCAL_FILES") == "True"

            # Get model path
            if model_files_local:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent.parent.parent
                model_files_path = project_root / "model_files" / short_name
                model_files_path = str(model_files_path.resolve())
            else:
                model_files_path = f"/app/model_files/{short_name}"

            logger.info(f"Attempting to load model from path: {model_files_path}")

            # Validate model directory and files
            if not os.path.exists(model_files_path):
                raise FileNotFoundError(
                    f"Model directory not found: {model_files_path}"
                )

            # Using this to check for required files
            required_files = model_config.get("required_files")
            missing_files = [
                f
                for f in required_files
                if not os.path.exists(os.path.join(model_files_path, f))
            ]
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")

            # Initializing the model based on model type
            model_type = model_config.get("type")
            if model_type == "text":
                self.initialize_text_model(model_files_path, **model_kwargs)
            elif model_type == "ner":
                self.initialize_ner_model(model_files_path, **model_kwargs)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            set_server_status("Model initialization FAILED")
            raise

    def initialize_ner_model(self, model_files_path, **model_kwargs):
        try:
            self._model = GLiNER.from_pretrained(
                model_files_path,
                local_files_only=True,
                device=self._device,
            )
            self._initialized = True
            logger.info("NER Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NER model: {e}")
            raise

    def initialize_text_model(self, model_files_path, **model_kwargs):
        try:
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
            self._initialized = True
            logger.info("Text Gen Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize text model: {e}")
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

    @property
    def device(self):
        return self._device
