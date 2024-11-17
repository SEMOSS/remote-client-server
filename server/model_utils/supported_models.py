SUPPORTED_MODELS = {
    "pixart": {
        "model_repo_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "short_name": "pixart",
        "type": "image",
        "required_files": ["model_index.json"],
        "use_flash_attention": False,
    },
    "phi-3-mini-128k-instruct": {
        "model_repo_id": "microsoft/Phi-3-mini-128k-instruct",
        "short_name": "phi-3-mini-128k-instruct",
        "type": "text",
        "required_files": ["config.json", "generation_config.json"],
        "use_flash_attention": False,
    },
    "gliner-multi-v2-1": {
        "model_repo_id": "urchade/gliner_multi-v2.1",
        "short_name": "gliner-multi-v2-1",
        "type": "ner",
        "required_files": ["gliner_config.json"],
        "use_flash_attention": False,
    },
    "gliner-large-v2-5": {
        "model_repo_id": "gliner-community/gliner_large-v2.5",
        "short_name": "gliner-large-v2-5",
        "type": "ner",
        "required_files": ["gliner_config.json"],
        "use_flash_attention": False,
    },
    "codellama-13b": {
        "model_repo_id": "codellama/CodeLlama-13b-hf",
        "short_name": "codellama-13b",
        "type": "text",
        "required_files": ["config.json", "generation_config.json"],
        "use_flash_attention": False,
    },
}
