import os
import json
import logging
import shutil
import subprocess
import time
from typing import Optional
from huggingface_hub import snapshot_download
from model_utils.model_config import get_model_config
from globals.globals import set_server_status

logger = logging.getLogger(__name__)


def verify_model_files(short_name: str, update_status: Optional[bool] = False) -> str:
    """
    Takes an expected path and checks the model files directory to see if the correct model files are present.
    Args:
        path (str): Path to the specific model files directory. IE: /app/model_files/pixart
    Returns:
        str: Message indicating if the model files are correct or not.
    """
    model_config = get_model_config()
    expected_model_class = model_config.get("expected_model_class")
    path = f"/app/model_files/{short_name}"
    model_index_path = os.path.join(path, "model_index.json")
    if not os.path.exists(path) or not os.listdir(path):
        message = "Model directory not found or empty."
        logger.info(message)
        if update_status:
            set_server_status(message)
        return message
    if not os.path.exists(model_index_path):
        message = "model_index.json not found."
        logger.info(message)
        if update_status:
            set_server_status(message)
        return message
    try:
        with open(model_index_path, "r") as f:
            model_info = json.load(f)

        currently_downloaded_model = model_info.get("_class_name")
        if currently_downloaded_model != expected_model_class:
            message = f"Existing model is listed as {currently_downloaded_model}, not {expected_model_class}."
            logger.info(message)
            if update_status:
                set_server_status(message)
            return message
        else:
            message = f"Correct model files for {expected_model_class} found."
            logger.info(message)
            if update_status:
                set_server_status(message)
            return message
    except json.JSONDecodeError:
        message = "Error parsing model_index.json."
        logger.error(message)
        return message


# This function is not currently in use but can be used to download model files dynamically.
# The main problem is how slow this operation is when run inside the container
def check_and_download_model_files():
    """
    Check if the model files exist and are for the correct model type.
    If not, delete existing files and download the correct ones.
    """
    set_server_status("Downloading model files...")

    LOCAL_MODEL_DIR = "/app/model_files/pixart"
    model_config = get_model_config()
    if not model_config:
        logger.error("Model configuration not found.")
        return

    model_repo_id = model_config.get("model_repo_id")
    expected_model_class = model_config.get("expected_model_class")

    model_index_path = os.path.join(LOCAL_MODEL_DIR, "model_index.json")

    # Check if directory exists and is not empty
    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        logger.info("Model directory not found or empty. Downloading model files...")
        # download_model_files(model_repo_id, LOCAL_MODEL_DIR)
        download_model_files_v2(model_repo_id, LOCAL_MODEL_DIR)
        return

    # Check if model_index.json exists
    if not os.path.exists(model_index_path):
        logger.info(
            "model_index.json not found. Clearing directory and downloading model files..."
        )
        clear_directory(LOCAL_MODEL_DIR)
        # download_model_files(model_repo_id, LOCAL_MODEL_DIR)
        download_model_files_v2(model_repo_id, LOCAL_MODEL_DIR)
        return

    # Parse model_index.json and check _class_name
    try:
        with open(model_index_path, "r") as f:
            model_info = json.load(f)

        currently_downloaded_model = model_info.get("_class_name")
        if currently_downloaded_model != expected_model_class:
            logger.info(
                f"Existing model is listed as {currently_downloaded_model}, not {expected_model_class}. Clearing directory and downloading correct model files..."
            )
            clear_directory(LOCAL_MODEL_DIR)
            # download_model_files(model_repo_id, LOCAL_MODEL_DIR)
            download_model_files_v2(model_repo_id, LOCAL_MODEL_DIR)
        else:
            logger.info(f"Correct model files for {expected_model_class} found.")
            set_server_status("ready")
    except json.JSONDecodeError:
        logger.error(
            "Error parsing model_index.json. Clearing directory and downloading model files..."
        )
        clear_directory(LOCAL_MODEL_DIR)
        # download_model_files(model_repo_id, LOCAL_MODEL_DIR)
        download_model_files_v2(model_repo_id, LOCAL_MODEL_DIR)


def clear_directory(directory):
    """
    Delete all contents of the specified directory.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")


def download_model_files(model_repo_id, LOCAL_MODEL_DIR):
    """
    Download the model files using snapshot_download.
    """
    snapshot_download(repo_id=model_repo_id, local_dir=LOCAL_MODEL_DIR)
    set_server_status("ready")
    logger.info("Model files downloaded successfully.")


def log_output(process, logger):
    for line in iter(process.stdout.readline, ""):
        logger.info(line.strip())


def download_model_files_v2(model_repo_id, LOCAL_MODEL_DIR):
    """
    Download the model files using the Hugging Face CLI with real-time logging and progress updates. This was implemented due to some issues with the snapshot_download function.
    """
    set_server_status("Downloading model files...")

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    command = f"huggingface-cli download {model_repo_id} --local-dir {LOCAL_MODEL_DIR} --local-dir-use-symlinks False"

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        last_log_time = time.time()
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    logger.info(output.strip())
                    last_log_time = current_time

        remaining_output, error_output = process.communicate()
        if remaining_output:
            logger.info(remaining_output.strip())
        if error_output:
            logger.error(error_output.strip())

        if process.returncode == 0:
            set_server_status("ready")
            logger.info("Model files downloaded successfully.")
        else:
            raise subprocess.CalledProcessError(
                process.returncode, command, error_output
            )

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading model files: {e}")
        logger.error(f"Error output: {e.output}")
        set_server_status("Error downloading model files.")
    except Exception as e:
        logger.error(f"Unexpected error during model download: {str(e)}")
        set_server_status("Error downloading model files.")
