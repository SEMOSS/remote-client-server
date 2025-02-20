from vllm import AsyncLLMEngine, AsyncEngineArgs
import logging
from typing import Optional
from gaas.model_manager.model_files_manager import ModelFilesManager

logger = logging.getLogger(__name__)


class VLLMManager:
    """Manages vLLM engine instance and configuration"""

    _instance = None
    _initialized = False
    _engine: Optional[AsyncLLMEngine] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize_engine(self, **kwargs):
        """Initialize vLLM engine with given configuration"""
        model_files_manager = ModelFilesManager()
        model_files_path = model_files_manager.get_model_files_path()
        try:
            engine_args = AsyncEngineArgs(
                model=model_files_path,
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
                max_num_batched_tokens=kwargs.get("max_num_batched_tokens", 4096),
                trust_remote_code=True,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            logger.info("vLLM engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            return False

    @property
    def engine(self):
        return self._engine
