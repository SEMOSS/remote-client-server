import os
import logging
from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from globals.globals import set_server_status
from model_utils.model_config import get_short_name

logger = logging.getLogger(__name__)


class AbstractTextGen(ABC):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.seed = torch.random.manual_seed(0)

        self._load_model()

    def _load_model(self):
        short_name = get_short_name()
        # Used in development mode if you are running outside of a docker container otherwise model files should be loaded in attached volume
        model_files_local = os.environ.get("LOCAL_FILES") == "True"
        if model_files_local:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_files_path = os.path.join(
                script_dir, "..", "..", "model_files", short_name
            )
        else:
            model_files_path = f"/app/model_files/{short_name}"
        logger.info(f"Loading the model from path: {model_files_path}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_files_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_files_path)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=tokenizer,
            )
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            set_server_status(f"{self.__class__.__name__} FAILED to initialize.")
            raise

    @abstractmethod
    def ask_model(
        self, prompt: str, temp: float, prob: float, max_tokens: int, **kwargs
    ):
        pass

    @abstractmethod
    def generate(
        self, task: str, temp: float, prob: float, max_tokens: float, **kwargs
    ):
        pass
