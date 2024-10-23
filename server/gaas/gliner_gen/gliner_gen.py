import os
import torch
from typing import List
import logging
from gliner import GLiNER
from globals.globals import set_server_status
from model_utils.model_config import get_short_name

logger = logging.getLogger(__name__)


class GlinerGen:
    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """
        Loads the GLiNER model from the model files path into memory.
        """
        short_name = get_short_name()
        model_files_local = os.environ.get("LOCAL_FILES") == "True"
        if model_files_local:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_files_path = os.path.join(
                script_dir, "..", "..", "..", "model_files", short_name
            )
            model_files_path = os.path.abspath(model_files_path)
        else:
            model_files_path = f"/app/model_files/{short_name}"

        try:
            self.model = GLiNER.from_pretrained(
                model_files_path, local_files_only=True, device=self.device
            )

        except Exception as e:
            logger.exception("Failed to initialize GLiNER: %s", e)
            set_server_status("GLiNER FAILED to initialize.")
            raise

    def generate(
        self,
        text: str,
        labels: List[str],
        **kwargs,
    ):
        """
        Generates entities using the GLiNER model.
        """
        try:
            entities = self.model.predict_entities(text, labels)
            return {"entities": entities}

        except Exception as e:
            logger.exception("Failed to generate output: %s", e)
            raise
