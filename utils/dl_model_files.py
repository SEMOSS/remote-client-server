from huggingface_hub import snapshot_download


def download_model_files():
    """_summary_
    This method is only for use in development when you want to download the model files locally.
    Useful for inspecting the expected model files or for running outside of a docker container.
    Use the `snapshot_download` function from the `huggingface_hub` library to download the model files locally.
    """
    # change this
    model_repo_id = "bytedance-research/UI-TARS-7B-DPO"
    # change this
    short_name = "ui-tars-7b-dpo"

    # don't change this
    local_model_dir = f"./model_files/{short_name}"

    snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)


if __name__ == "__main__":
    download_model_files()
