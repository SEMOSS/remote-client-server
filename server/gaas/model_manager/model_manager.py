import os
import logging
import time
import torch
import subprocess
import threading
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoProcessor,
    AutoImageProcessor,
    AutoConfig,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from gliner import GLiNER
from model_utils.model_config import get_model_config
from gaas.tokenizer.tokenizer import Tokenizer
from gaas.model_manager.model_files_manager import ModelFilesManager
from globals.globals import set_server_status
from gaas.model_manager.loaders.vision_loader import VisionModelLoader

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
            cls._instance._device = cls._instance._get_device_config()
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_device_config(self):
        """Get device configuration from environment variables."""
        device_str = os.environ.get("MODEL_DEVICE", "")
        if not device_str:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(
                    f"CUDA specified but GPU not available, falling back to CPU"
                )
                return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            return torch.device(device_str)
        except Exception as e:
            logger.warning(
                f"Invalid device '{device_str}' specified, falling back to default. Error: {e}"
            )
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        start_time = time.time()

        gpu_monitor = None
        hasGPU = torch.cuda.is_available()
        if hasGPU:
            gpu_monitor = GPUMonitor()
            gpu_monitor.start()
        else:
            logger.info("No GPU monitoring since GPU is not available")

        try:
            t0 = time.time()
            flash_attn_available = self._check_flash_attention()
            if flash_attn_available:
                logger.info(
                    f"Flash attention check took: {time.time() - t0:.2f} seconds"
                )

            t0 = time.time()
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(model_files_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            logger.info(
                f"Model directory analysis - Files: {file_count}, Total size: {total_size / (1024*1024):.2f} MB"
            )
            logger.info(f"Directory analysis took: {time.time() - t0:.2f} seconds")

            try:
                if flash_attn_available:
                    t0 = time.time()
                    model_files_manager = ModelFilesManager()
                    model_uses_flash_attn = (
                        model_files_manager.analyze_flash_attention_compatibility()
                    )
                    logger.info(
                        f"Flash attention compatibility analysis took: {time.time() - t0:.2f} seconds"
                    )

                    if model_uses_flash_attn:
                        logger.info("Attempting to load model with Flash Attention 2")
                        t0 = time.time()
                        try:
                            # Trying Flash Attention 2 first
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                            model = model_class.from_pretrained(
                                model_files_path, **model_kwargs
                            )
                            logger.info(
                                f"Model loading with Flash Attention 2 took: {time.time() - t0:.2f} seconds"
                            )

                            if hasGPU:
                                gpu_monitor.stop()
                                gpu_monitor.join()
                            if gpu_monitor and gpu_monitor.readings:
                                max_memory = max(
                                    float(r.split(",")[1]) for r in gpu_monitor.readings
                                )
                                max_util = max(
                                    float(r.split(",")[3]) for r in gpu_monitor.readings
                                )
                                logger.info(
                                    f"Peak GPU memory during FA2 load: {max_memory:.2f} MB"
                                )
                                logger.info(
                                    f"Peak GPU utilization during FA2 load: {max_util:.2f}%"
                                )
                            return model
                        except Exception as e:
                            logger.warning(
                                f"Failed to load with Flash Attention 2: {e}"
                            )
                            try:
                                # Fall back to Flash Attention 1
                                logger.info(
                                    "Attempting to load model with Flash Attention 1"
                                )
                                t0 = time.time()
                                model_kwargs["attn_implementation"] = "flash_attention"
                                model = model_class.from_pretrained(
                                    model_files_path, **model_kwargs
                                )
                                logger.info(
                                    f"Model loading with Flash Attention 1 took: {time.time() - t0:.2f} seconds"
                                )

                                if hasGPU:
                                    gpu_monitor.stop()
                                    gpu_monitor.join()
                                if gpu_monitor and gpu_monitor.readings:
                                    max_memory = max(
                                        float(r.split(",")[1])
                                        for r in gpu_monitor.readings
                                    )
                                    max_util = max(
                                        float(r.split(",")[3])
                                        for r in gpu_monitor.readings
                                    )
                                    logger.info(
                                        f"Peak GPU memory during FA1 load: {max_memory:.2f} MB"
                                    )
                                    logger.info(
                                        f"Peak GPU utilization during FA1 load: {max_util:.2f}%"
                                    )
                                return model
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load with Flash Attention: {e}"
                                )
                                logger.info("Falling back to standard attention")
                    else:
                        logger.warning(
                            "Flash attention available but model does not use it.. Loading model with standard attention"
                        )

                # Standard loading without flash attention
                t0_total = time.time()
                logger.info("Starting standard model loading...")

                # Track config loading
                t0 = time.time()
                config = AutoConfig.from_pretrained(
                    model_files_path, trust_remote_code=True
                )
                logger.info(f"Config loading took: {time.time() - t0:.2f} seconds")

                # Track actual model loading
                t0 = time.time()
                model_kwargs.pop("attn_implementation", None)
                model = model_class.from_pretrained(
                    model_files_path, config=config, **model_kwargs
                )
                logger.info(
                    f"Actual model loading took: {time.time() - t0:.2f} seconds"
                )

                total_load_time = time.time() - t0_total
                logger.info(
                    f"Standard model loading total time: {total_load_time:.2f} seconds"
                )
                if hasGPU:
                    gpu_monitor.stop()
                    gpu_monitor.join()

                # Log GPU monitoring summary
                if gpu_monitor and gpu_monitor.readings:
                    max_memory = max(
                        float(r.split(",")[1]) for r in gpu_monitor.readings
                    )
                    max_util = max(float(r.split(",")[3]) for r in gpu_monitor.readings)
                    logger.info(f"Peak GPU memory during load: {max_memory:.2f} MB")
                    logger.info(f"Peak GPU utilization during load: {max_util:.2f}%")

                # Log memory stats right after loading
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_reserved = torch.cuda.memory_reserved() / 1024**2
                    logger.info(
                        f"GPU Memory after loading - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB"
                    )

                return model

            except Exception as e:
                if hasGPU:
                    gpu_monitor.stop()
                    gpu_monitor.join()
                logger.error(f"Failed to initialize model: {e}")
                raise

        except Exception as e:
            if hasGPU:
                gpu_monitor.stop()
                gpu_monitor.join()
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
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                # "torch_dtype": torch.float16,
                # "use_safetensors": True,
                "use_auth_token": False,
                "local_files_only": False,
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
            elif model_type == "vision-embed":
                return self.initialize_image_embedding_model(
                    model_files_path, **model_kwargs
                )
            elif model_type == "vision":
                return self.initialize_vision_model(model_files_path, **model_kwargs)
            elif model_type == "emotion":
                return self.initialize_emotion_model(model_files_path)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            set_server_status("Model initialization FAILED")
            raise

    def initialize_embedding_model(self, model_files_path, **model_kwargs):
        """Initialize an embedding model with optimized loading parameters."""
        try:
            total_start = time.time()
            logger.info(
                f"Starting embedding model initialization on device: {self._device}"
            )

            model_kwargs.update(
                {
                    "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
                    "low_cpu_mem_usage": True,  # More memory efficient loading
                    "torch_dtype": (
                        torch.float16
                        if str(self._device).startswith("cuda")
                        else torch.float32
                    ),  # Use half precision during loading
                    "use_safetensors": True,  # Faster loading if safetensors are available
                    "use_auth_token": False,  # Ensure we're not trying to download
                    "local_files_only": False,  # Prevent any download attempts
                    "trust_remote_code": True,
                }
            )

            logger.info(f"Using optimized model kwargs: {model_kwargs}")

            t0 = time.time()
            self._model = self.initialize_model_with_flash_attention(
                AutoModel,
                model_files_path,
                **model_kwargs,
            )
            logger.info(f"Model initialization took: {time.time() - t0:.2f} seconds")

            # Move to GPU efficiently
            t0 = time.time()
            logger.info(f"Moving model to device: {self._device}")
            self._model = self._model.to(self._device, non_blocking=True)
            logger.info(f"Moving to device took: {time.time() - t0:.2f} seconds")

            t0 = time.time()
            logger.info("Setting model to eval mode...")
            self._model.eval()
            logger.info(f"Setting eval mode took: {time.time() - t0:.2f} seconds")

            t0 = time.time()
            logger.info("Initializing tokenizer...")
            self._tokenizer = Tokenizer().tokenizer
            logger.info(
                f"Tokenizer initialization took: {time.time() - t0:.2f} seconds"
            )

            logger.info(f"Final model device: {next(self._model.parameters()).device}")

            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f} MB")
                logger.info(f"GPU Memory reserved: {memory_reserved:.2f} MB")

            self._initialized = True
            total_time = time.time() - total_start
            logger.info(
                f"Total embedding model initialization took: {total_time:.2f} seconds"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def initialize_image_embedding_model(self, model_files_path, **model_kwargs):
        """Initialize an image embedding model."""
        try:
            logger.info(f"Initializing image embedding model on device: {self._device}")

            if "device_map" in model_kwargs:
                del model_kwargs["device_map"]

            self._model = self.initialize_model_with_flash_attention(
                AutoModel,
                model_files_path,
                **model_kwargs,
            )

            self._model = self._model.to(self._device)
            self._model.eval()

            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_files_path, trust_remote_code=True
                )
            except Exception:
                self._processor = AutoImageProcessor.from_pretrained(
                    model_files_path, trust_remote_code=True
                )

            logger.info(
                f"Model device after initialization: {next(self._model.parameters()).device}"
            )

            self._initialized = True
            logger.info("Image Embedding Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize image embedding model: {e}")
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
            # First try to load the config to check the architecture
            config = AutoConfig.from_pretrained(
                model_files_path, trust_remote_code=True
            )

            # Check if it's a VL (Vision-Language) model
            if hasattr(config, "architectures") and any(
                "VL" in arch for arch in config.architectures
            ):
                from transformers import AutoModelForVision2Seq

                logger.info(
                    "Detected Vision-Language model, using AutoModelForVision2Seq"
                )
                self._model = self.initialize_model_with_flash_attention(
                    AutoModelForVision2Seq,
                    model_files_path,
                    **model_kwargs,
                )
            else:
                # Standard text model loading
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

    def initialize_vision_model(self, model_files_path, **model_kwargs):
        """Initialize a vision or vision-language model."""
        try:
            vision_loader = VisionModelLoader(self)
            success = vision_loader.load_vision_model(model_files_path, model_kwargs)

            if success:
                self._model = vision_loader.model
                self._processor = vision_loader.processor
                self._tokenizer = vision_loader.tokenizer
                self._initialized = True
                logger.info("Vision/Vision-Language Model loaded successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise

    def initialize_emotion_model(self, model_files_path):
        # """Initialize the emotion classification model."""
        try:
            logger.info(f"Initializing Emotion Model on device: {self._device}")
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_files_path)
            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_files_path
            ).to(self._device)
            self._model.eval()

            logger.info("Emotion Model initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Emotion model: {e}")
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

    @property
    def processor(self):
        return self._processor


class GPUMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.readings = []

    def run(self):
        while self.running:
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                )
                reading = output.decode("utf-8").strip()
                self.readings.append(reading)
                logger.info(f"GPU Status during load: {reading}")
            except Exception as e:
                logger.error(f"Failed to get GPU stats: {e}")
            time.sleep(1)

    def stop(self):
        self.running = False
