import os
import torch
from typing import Any, Dict, List
import logging
import random
import string
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
        entities: List[str],
        mask_entities: List[str] = [],
        **kwargs,
    ):
        """
        Generates entities using the GLiNER model.
        """
        try:
            response = self.model.predict_entities(text, entities)
        except Exception as e:
            logger.exception("Failed to generate output: %s", e)
            return {
                "status": "error",
                "output": [],
                "raw_output": [],
                "mask_values": {},
                "input": text,
                "entities": [],
            }

        masked_output = self._mask_entities(text, response, mask_entities)

        return {
            "status": "success",
            "output": masked_output["masked_text"],
            "raw_output": response,
            "mask_values": masked_output["mask_values"],
            "input": text,
            "entities": entities,
        }

    def _generate_mask(self, length: int = 6) -> str:
        """Generate a random mask string."""
        random_str = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=length)
        )
        return f"m_{random_str}"

    def _mask_entities(
        self, text: str, entities: List[Dict[str, Any]], mask_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Mask entities in the text based on the mask_entities list.

        Args:
            text (str): Original text
            entities (List[Dict[str, Any]]): Detected entities
            mask_entities (List[str]): List of entity types to mask

        Returns:
            Dict[str, Any]: Dictionary containing masked text and mapping
        """
        entities = sorted(entities, key=lambda x: x["start"], reverse=True)

        mask_values = {}
        new_text = text

        for entity in entities:
            if entity["label"] in mask_entities:
                orig_text = entity["text"]
                start = entity["start"]
                end = entity["end"]

                # Generate or retrieve mask
                if orig_text in mask_values:
                    mask_text = mask_values[orig_text]
                else:
                    mask_text = self._generate_mask()
                    mask_values[orig_text] = mask_text
                    mask_values[mask_text] = orig_text

                new_text = new_text[:start] + mask_text + new_text[end:]

        return {"masked_text": new_text, "mask_values": mask_values}
