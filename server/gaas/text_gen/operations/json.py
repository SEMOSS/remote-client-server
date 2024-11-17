import logging
from typing import Any, Dict, Optional
import guidance
import json
from gaas.text_gen.abstract_text_gen import AbstractTextGen
import torch

logger = logging.getLogger(__name__)


class JSON(AbstractTextGen):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model.to(torch.float16)
        self.guidance_model = guidance.models.Transformers(
            model=self.model, tokenizer=self.tokenizer, echo=False, dtype=torch.float16
        )

    def generate(
        self, prompt: str, json_schema: str, context: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Convert text data to JSON format based on a provided schema.

        Args:
            text_data: The input text to convert to JSON
            json_schema: A JSON schema string defining the expected output structure
            context: Custom context/prompt to guide the JSON conversion

        Returns:
            Dict containing:
                - input: Original text data
                - schema: Original schema
                - output: Generated JSON or error message
        """
        response = {"input": prompt, "schema": json_schema, "output": None}

        # Parse and validate JSON schema
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
            # Set up conversion context
            default_context = (
                "You are an expert who converts English text to JSON. "
                "When data is not available, use -111 as default value. "
                "Convert the following text to JSON:"
            )
            conversion_prompt = f"{context or default_context}\n{prompt}"

            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Generate JSON using guidance
                json_output = self.guidance_model + conversion_prompt
                json_output += guidance.json(
                    name="generated_object", schema=parsed_schema
                )

            # Extract and validate generated JSON
            generated_json = json_output.get("generated_object")
            if not generated_json:
                raise ValueError("No JSON output generated")

            response["output"] = generated_json

        except Exception as e:
            logger.error(f"Error in JSON generation process: {str(e)}")
            response["output"] = f"error: JSON generation failed - {str(e)}"

        return response
