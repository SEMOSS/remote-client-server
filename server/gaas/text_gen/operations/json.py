import logging
from typing import Optional
import guidance
import json
import torch
from gaas.model_manager.model_manager import ModelManager

logger = logging.getLogger(__name__)


class JSON:
    """Example of a text generation class using the shared model manager."""

    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.pipe = model_manager.pipe
        self.device = model_manager.device

        # JSON-specific initialization
        self.model.to(torch.float16)
        self.guidance_model = guidance.models.Transformers(
            model=self.model, tokenizer=self.tokenizer, echo=False, dtype=torch.float16
        )

    def generate(
        self, prompt: str, json_schema: str, context: Optional[str] = None, **kwargs
    ):
        response = {"input": prompt, "schema": json_schema, "output": None}

        try:
            parsed_schema = json.loads(json_schema)
            if not isinstance(parsed_schema, dict):
                logger.error("JSON schema must be an object")
                response["output"] = "error: Invalid JSON schema. Must be an object"
                return response
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON schema format: {str(e)}")
            response["output"] = f"error: invalid JSON schema - {str(e)}"
            return response

        try:
            default_context = (
                "You are an expert who converts English text to JSON. "
                "When data is not available, use -111 as default value. "
                "Convert the following text to JSON:"
            )
            conversion_prompt = f"{context or default_context}\n{prompt}"

            with torch.amp.autocast("cuda", dtype=torch.float16):
                json_output = self.guidance_model + conversion_prompt
                json_output += guidance.json(
                    name="generated_object", schema=parsed_schema
                )

            generated_json = json_output.get("generated_object")
            if not generated_json:
                raise ValueError("No JSON output generated")

            response["output"] = generated_json

        except Exception as e:
            logger.error(f"Error in JSON generation process: {str(e)}")
            response["output"] = f"error: JSON generation failed - {str(e)}"

        return response
