from huggingface_hub import snapshot_download


def download_model_files():
    """_summary_
    Use the `snapshot_download` function from the `huggingface_hub` library to download the model files locally.
    This prevents the Docker container from having to download the model files every time the container is started.
    """
    model_repo_id = "urchade/gliner_multi-v2.1"

    local_model_dir = "./model_files/gliner-multi-v2.1"

    snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)


if __name__ == "__main__":
    download_model_files()
