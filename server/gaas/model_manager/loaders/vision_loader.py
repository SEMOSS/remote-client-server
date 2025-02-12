import logging
import torch
import gc
import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class VisionModelLoader:
    def __init__(self, model_manager):
        self.manager = model_manager
        self._device = model_manager.device
        self._model = None
        self._processor = None
        self._tokenizer = None

    def load_vision_model(self, model_files_path, model_kwargs):
        """Initialize a vision or vision-language model."""
        try:
            logger.info(f"Initializing vision model on device: {self._device}")

            model_kwargs.update(
                {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "offload_folder": "offload",
                }
            )

            config = AutoConfig.from_pretrained(
                model_files_path, trust_remote_code=True
            )
            logger.info(f"Model type from config: {config.model_type}")

            if torch.cuda.is_available():
                self._setup_gpu_memory(model_kwargs)

            self._load_model_based_on_type(config, model_files_path, model_kwargs)
            self._load_processor(model_files_path)
            self._cleanup_memory()

            self._log_memory_stats()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise

    def _setup_gpu_memory(self, model_kwargs):
        """Configure GPU memory settings with better memory management."""
        if not torch.cuda.is_available():
            return

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        inference_buffer = min(gpu_memory * 0.25, 4)  # 25% or 4GB, whichever is smaller
        model_memory = gpu_memory - inference_buffer

        torch.cuda.empty_cache()
        gc.collect()

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,"
            "expandable_segments:True,"
            "garbage_collection_threshold:0.6"
        )

        model_kwargs.update(
            {
                "max_memory": {0: f"{model_memory:.0f}GiB"},
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
        )

        logger.info(f"GPU Memory Configuration:")
        logger.info(f"Total GPU Memory: {gpu_memory:.2f}GB")
        logger.info(f"Reserved for Inference: {inference_buffer:.2f}GB")
        logger.info(f"Available for Model: {model_memory:.2f}GB")

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"Initial GPU Memory State:")
        logger.info(f"Allocated: {allocated:.2f}GB")
        logger.info(f"Reserved: {reserved:.2f}GB")

    def _cleanup_memory(self):
        """Enhanced memory cleanup between operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            torch.cuda.synchronize()

            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Memory after cleanup:")
            logger.info(f"Allocated: {allocated:.2f}GB")
            logger.info(f"Reserved: {reserved:.2f}GB")

    def _load_model_based_on_type(self, config, model_files_path, model_kwargs):
        if "qwen2_vl" in config.model_type.lower():
            self._load_qwen_model(model_files_path, model_kwargs)
        elif hasattr(config, "architectures") and any(
            "VL" in arch for arch in config.architectures
        ):
            self._load_vision_language_model(model_files_path, model_kwargs)
        else:
            self._load_standard_vision_model(model_files_path, model_kwargs)

    def _load_qwen_model(self, model_files_path, model_kwargs):
        """Initialize Qwen2VL model with its tokenizer"""

        logger.info("Initializing Qwen2VL model")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_files_path, **model_kwargs
        )

        logger.info("Initializing Qwen2VL tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_files_path, trust_remote_code=True, padding_side="left"
        )

        logger.info(
            f"Tokenizer loaded successfully. Vocab size: {len(self._tokenizer)}"
        )

    def _load_vision_language_model(self, model_files_path, model_kwargs):
        logger.info("Loading as Vision-Language model using AutoModelForVision2Seq")
        self._model = self.manager.initialize_model_with_flash_attention(
            AutoModelForVision2Seq, model_files_path, **model_kwargs
        )

    def _load_standard_vision_model(self, model_files_path, model_kwargs):
        logger.info("Loading as standard vision model using AutoModelForCausalLM")
        self._model = self.manager.initialize_model_with_flash_attention(
            AutoModelForCausalLM, model_files_path, **model_kwargs
        )

    def _load_processor(self, model_files_path):
        logger.info("Attempting to load processor...")
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_files_path, trust_remote_code=True
            )
            logger.info("Successfully loaded AutoProcessor")
        except Exception as processor_error:
            logger.error(f"Error loading processor: {processor_error}")
            raise

    def _log_memory_stats(self):
        if torch.cuda.is_available():
            device_id = next(self._model.parameters()).device
            logger.info(f"Model loaded on device: {device_id}")
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                f"After initialization - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
            )

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def tokenizer(self):
        return self._tokenizer
