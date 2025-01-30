import os
import logging
from pathlib import Path
from model_utils.model_config import get_model_config
import json

logger = logging.getLogger(__name__)


class ModelFilesManager:
    def __init__(self):
        self.model_config = get_model_config()
        self.model = self.model_config.get("model")
        self.model_type = self.model_config.get("type")
        self.use_local_files = os.environ.get("LOCAL_FILES") == "True"

    def get_model_files_path(self):
        """Get the path to the current model's model files and verify its existence."""
        try:
            if self.use_local_files:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent.parent.parent
                model_files_path = project_root / "model_files" / self.model
            else:
                model_files_path = Path(f"/app/model_files/{self.model}")

            # Verify the path exists and is a directory
            if not model_files_path.exists():
                raise FileNotFoundError(
                    f"Model directory does not exist: {model_files_path}"
                )
            if not model_files_path.is_dir():
                raise NotADirectoryError(
                    f"Model path is not a directory: {model_files_path}"
                )

            return str(model_files_path.resolve())

        except Exception as e:
            logger.error(f"Failed to get model files path: {str(e)}")
            raise FileNotFoundError(f"Could not access model directory: {str(e)}")

    def get_config_json_path(self):
        """Get the path to the current model's config.json file."""
        if self.model_type == "ner":
            file_name = "gliner_config.json"
        else:
            file_name = "config.json"
        model_files_path = self.get_model_files_path()
        config_path = Path(model_files_path) / file_name
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None
        return str(config_path.resolve())

    def analyze_flash_attention_compatibility(self) -> bool:
        """Analyze model config for Flash Attention compatibility."""
        config_json_path = self.get_config_json_path()
        if not config_json_path:
            logger.info("Config file not found.. Skipping Flash Attention analysis")
            return False

        with open(config_json_path) as f:
            config = json.load(f)

        # 1. Explicit Flash Attention flags
        explicit_flags = [
            config.get("use_flash_attn"),
            config.get("use_flash_attention"),
            config.get("attention_implementation") == "flash_attention",
            config.get("attention_implementation") == "flash_attention_2",
        ]
        if any(explicit_flags):
            logger.info(
                "Explicit Flash Attention flag found.. Enabling Flash Attention"
            )
            return True

        # 2. Check for known architectures that commonly use Flash Attention
        model_type = config.get("model_type", "").lower()
        architectures = [arch.lower() for arch in config.get("architectures", [])]
        flash_common_architectures = {
            "phi",
            "mistral",
            "llama",
            "falcon",
            "mixtral",
            "qwen",
            "yi",
            "gemma",
            "mamba",
        }

        if model_type in flash_common_architectures:
            logger.info(
                f"Model type {model_type} typically uses Flash Attention.. Enabling Flash Attention"
            )
            return True

        arch_match = any(
            any(fa in arch for fa in flash_common_architectures)
            for arch in architectures
        )
        if arch_match:
            logger.info(
                f"Model architecture {architectures} typically uses Flash Attention.. Enabling Flash Attention"
            )
            return True

        # 3. Check for Flash Attention architectural patterns
        pattern_indicators = {
            "Modern KV Cache": config.get("num_key_value_heads") is not None,
            "Sliding Window": config.get("sliding_window") is not None,
            "No Attention Bias": config.get("attention_bias") is False,
            "RoPE Scaling": "rope_scaling" in config,
            "Rotary Embeddings": config.get("position_embedding_type") == "rotary",
            "GQA Support": config.get(
                "num_key_value_heads", config.get("num_attention_heads")
            )
            != config.get("num_attention_heads"),
            "Large Context": config.get("max_position_embeddings", 0) > 8192,
        }

        # Count Flash Attention patterns == True
        pattern_count = sum(pattern_indicators.values())
        if pattern_count >= 3:  # If find >= 3 indicators
            matching_patterns = [k for k, v in pattern_indicators.items() if v]
            logger.info(
                f"Found multiple Flash Attention patterns: {', '.join(matching_patterns)} .. Enabling Flash Attention"
            )
            return True

        # 4. Check for custom implementations
        custom_indicators = [
            "flash" in str(config.get("auto_map", {})).lower(),
            any(
                "flash" in str(module).lower()
                for module in config.get("custom_modules", [])
            ),
            "flash" in str(config.get("attention_module", "")).lower(),
        ]
        if any(custom_indicators):
            logger.info(
                "Custom Flash Attention implementation found.. Enabling Flash Attention"
            )
            return True

        # 5. Model-specific optimizations
        if (
            config.get("model_type") == "xlm-roberta"
            and config.get("position_embedding_type") == "rotary"
        ):
            logger.info(
                "XLM-Roberta with Rotary Position Embeddings typically uses Flash Attention.. Enabling Flash Attention"
            )
            return True

        # No clear indicators found
        logger.info(
            "No clear Flash Attention compatibility indicators found.. Disabling Flash Attention"
        )
        return False
