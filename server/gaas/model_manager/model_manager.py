import logging
import torch
from transformers import AutoModelForCausalLM, AutoModel, pipeline
from gliner import GLiNER
from model_utils.model_config import get_model_config
from gaas.tokenizer.tokenizer import Tokenizer
from gaas.model_manager.model_files_manager import ModelFilesManager
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
        try:
            import flash_attn  # type: ignore

            logger.info("Flash attention is available.")
            return True
        except ImportError:
            logger.warning("Flash attention is not available.")
            return False

    def initialize_model_with_flash_attention(
        self, model_class, model_files_path: str, **model_kwargs
    ):
        """Initialize a model with proper Flash Attention settings based on analysis"""
        # Check if flash_attn is available on container
        flash_attn_available = self._check_flash_attention()

        try:
            if flash_attn_available:
                # Check if model uses flash attention
                model_files_manager = ModelFilesManager()
                model_uses_flash_attn = (
                    model_files_manager.analyze_flash_attention_compatibility()
                )

                if model_uses_flash_attn:
                    logger.info("Attempting to load model with Flash Attention 2")
                    try:
                        # Trying Flash Attention 2 first
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        return model_class.from_pretrained(
                            model_files_path, **model_kwargs
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load with Flash Attention 2: {e}")
                        try:
                            # Fall back to Flash Attention 1
                            logger.info(
                                "Attempting to load model with Flash Attention 1"
                            )
                            model_kwargs["attn_implementation"] = "flash_attention"
                            return model_class.from_pretrained(
                                model_files_path, **model_kwargs
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load with Flash Attention: {e}")
                            logger.info("Falling back to standard attention")
                else:
                    logger.warning(
                        "Flash attention available but model does not use it.. Loading model with standard attention"
                    )

            # Remove any flash attention config if we get here
            model_kwargs.pop("attn_implementation", None)
            return model_class.from_pretrained(model_files_path, **model_kwargs)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def initialize_model(self):
        """Initialize the model, tokenizer, and pipeline if not already initialized."""
        if self._initialized:
            logger.debug("Model already initialized, skipping initialization")
            return

        logger.info("Initializing model!")
        try:
            model_files_manager = ModelFilesManager()
            model_files_path = model_files_manager.get_model_files_path()
            model_kwargs = {
                "device_map": "cuda",
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
            }

            model_config = get_model_config()
            # Initializing the model based on model type
            model_type = model_config.get("type")
            if model_type == "text":
                return self.initialize_text_model(model_files_path, **model_kwargs)
            elif model_type == "ner":
                return self.initialize_ner_model(model_files_path, **model_kwargs)
            elif model_type == "embed":
                return self.initialize_embedding_model(model_files_path, **model_kwargs)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            set_server_status("Model initialization FAILED")
            raise

    def initialize_embedding_model(self, model_files_path, **model_kwargs):
        """Initialize an embedding model."""
        try:
            logger.info(f"Initializing embedding model on device: {self._device}")

            if "device_map" in model_kwargs:
                del model_kwargs["device_map"]

            self._model = self.initialize_model_with_flash_attention(
                AutoModel,
                model_files_path,
                **model_kwargs,
            )

            self._model = self._model.to(self._device)
            self._model.eval()
            self._tokenizer = Tokenizer().tokenizer

            logger.info(
                f"Model device after initialization: {next(self._model.parameters()).device}"
            )

            self._initialized = True
            logger.info("Embedding Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
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
            self._model = self.initialize_model_with_flash_attention(
                AutoModelForCausalLM,
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
