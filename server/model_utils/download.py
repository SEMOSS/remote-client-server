import os
import json
import logging
import shutil
import subprocess
import time
from typing import Set
from model_utils.model_config import get_model_config
from globals.globals import set_server_status
from huggingface_hub import snapshot_download, HfApi

logger = logging.getLogger(__name__)


def get_all_files(directory: str) -> Set[str]:
    """Get all files in directory and subdirectories."""
    files = set()
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.add(os.path.join(root, filename))
    return files


def check_min_file_size(filepath: str, min_size_kb: int = 1) -> bool:
    """Check if file exists and is larger than minimum size."""
    try:
        size_kb = os.path.getsize(filepath) / 1024
        return size_kb >= min_size_kb
    except (OSError, FileNotFoundError):
        return False


def check_file_size(filepath: str, directory: str, expected_sizes: list) -> bool:
    """Check if file exists and matches expected size from HF. Skip metadata and cache files"""
    if filepath.endswith(".metadata") or ".cache" in filepath:
        return True
    try:
        size = os.path.getsize(filepath)
        filepath = filepath.replace("\\", "/")
        directory = directory.replace("\\", "/")
        filename = filepath.removeprefix(directory + "/")
        if not expected_sizes[filename]:
            # Return true since no expected size from HF exists
            logger.warning(f"{filename} did not have a expected size from HF")
            return True
        if size == expected_sizes[filename]:
            return True
        else:
            logger.warning(
                f"File size mismatch for {filepath}. Wanted {expected_sizes[filename]}, but got {size}"
            )
            return False

    except (OSError, FileNotFoundError):
        return False


def get_expected_file_sizes(repo_id: str) -> dict:
    """Retrieve expected file sizes from Hugging Face repository metadata."""
    file_sizes = {}
    try:
        model_api = HfApi()
        model_info = model_api.model_info(repo_id, files_metadata=True)
        for repo_file in model_info.siblings:
            file_sizes[repo_file.rfilename] = repo_file.size
    except Exception as e:
        logger.error(f"Error retrieving file metadata for {repo_id}: {str(e)}")
    return file_sizes


def verify_download_completion(model_dir: str, expected_sizes: list) -> bool:
    """
    Verify if all model files are completely downloaded and valid.

    This function checks for:
    1. Common model format files (safetensors, bin, model, json)
    2. Essential configuration files
    3. File size validation for key files
    4. Basic file integrity

    Args:
        model_dir: Path to the directory containing the model files

    Returns:
        bool: True if verification passes, False otherwise
    """

    try:
        # 1. Check if directory exists and is not empty
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            logger.error(
                f"Model directory {model_dir} does not exist or is not a directory"
            )
            return False

        all_files = get_all_files(model_dir)
        if not all_files:
            logger.error(f"No files found in {model_dir}")
            return False

        # 2. Check for essential files by extension
        essential_extensions = {
            ".json",  # Config files
            ".txt",  # README, license, etc.
            ".md",  # Documentation
        }

        model_extensions = {
            ".safetensors",  # Common model format
            ".bin",  # Binary model files
            ".model",  # Another common model format
            ".onnx",  # ONNX format
            ".pth",  # PyTorch format
        }

        # Check if we have at least one model file
        has_model_file = any(
            any(f.endswith(ext) for ext in model_extensions) for f in all_files
        )

        if not has_model_file:
            logger.error("No model files found with standard extensions")
            return False

        # 3. Check for config files
        has_config = any(
            any(f.endswith(ext) for ext in essential_extensions) for f in all_files
        )

        if not has_config:
            logger.error("No configuration files found")
            return False

        # 4. Validate file sizes
        checked_files = [
            f for f in all_files if check_min_file_size(f, min_size_kb=100)
        ]
        if not checked_files:
            logger.error(
                "No files larger than 100KB found - possible incomplete download"
            )
            return False

        unmatched_files = [
            f.removeprefix(model_dir)
            for f in all_files
            if not check_file_size(f, model_dir, expected_sizes)
        ]

        if unmatched_files:
            logger.error(
                "One or more files didn't match expected size from HF - possible incomplete download"
            )
        else:
            logger.info("All files are equal to the size of respective file in HF repo")

        # 5. Try to load and validate config.json if it exists
        config_files = [f for f in all_files if f.endswith("config.json")]
        if config_files:
            try:
                with open(config_files[0], "r") as f:
                    config = json.load(f)
                if not isinstance(config, dict):
                    logger.error("config.json is not a valid JSON object")
                    return False
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error reading config.json: {str(e)}")
                return False

        # 6. Check for any .git* files that might indicate incomplete LFS download
        git_lfs_patterns = [".gitattributes", ".git/lfs"]
        incomplete_lfs = any(
            any(pattern in f for pattern in git_lfs_patterns) for f in all_files
        )
        if incomplete_lfs:
            logger.warning("Found Git LFS files - download might be incomplete")

        logger.info("Model verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return False


def download_with_git_lfs(repo_id: str, local_dir: str) -> bool:
    """
    Download model using git-lfs with explicit LFS file handling.
    """
    try:
        git_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Downloading from: {git_url}")

        if os.path.exists(local_dir):
            logger.info(f"Removing existing directory: {local_dir}")
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Created directory: {local_dir}")

        git_env = os.environ.copy()
        git_env.update(
            {
                "GIT_LFS_SKIP_SMUDGE": "1",  # Skip LFS files during clone
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_PROGRESS": "true",
                "GIT_TRACE": "1",
                "HOME": "/root",
                "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
            }
        )

        logger.info("Performing initial clone without LFS files...")
        clone_cmd = [
            "git",
            "clone",
            "--progress",
            "--depth",
            "1",
            "--single-branch",
            "--no-checkout",
            git_url,
            ".",
        ]

        clone_process = subprocess.run(
            clone_cmd,
            cwd=local_dir,
            env=git_env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Clone output: {clone_process.stdout}")
        if clone_process.stderr:
            logger.info(f"Clone stderr: {clone_process.stderr}")

        # initialize LFS in the repo
        logger.info("Initializing LFS in repository...")
        subprocess.run(
            ["git", "lfs", "install", "--local"], cwd=local_dir, env=git_env, check=True
        )

        # Checkout the files first
        logger.info("Checking out files...")
        subprocess.run(
            ["git", "checkout", "HEAD"], cwd=local_dir, env=git_env, check=True
        )

        # Explicitly fetch LFS files
        logger.info("Fetching LFS files...")
        lfs_fetch = subprocess.run(
            ["git", "lfs", "fetch", "--all"],
            cwd=local_dir,
            env=git_env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"LFS fetch output: {lfs_fetch.stdout}")

        # Checkout LFS files
        logger.info("Checking out LFS files...")
        lfs_checkout = subprocess.run(
            ["git", "lfs", "checkout"],
            cwd=local_dir,
            env=git_env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"LFS checkout output: {lfs_checkout.stdout}")

        # List files and their sizes
        logger.info("Checking downloaded files...")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                logger.info(f"File: {file_path}, Size: {size} bytes")

        if verify_download_completion(local_dir):
            logger.info("Download verified successfully")
            return True
        else:
            logger.error("Download verification failed")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(e.cmd)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during git-lfs download: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False


def download_model_files(
    model_repo_id: str, local_model_dir: str, max_retries: int = 2
) -> bool:
    """
    Download using git-lfs with retry logic.
    """
    set_server_status("Downloading model files...")

    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1} of {max_retries}")

            if download_with_git_lfs(model_repo_id, local_model_dir):
                logger.info("Model files downloaded and verified successfully")
                set_server_status("ready")
                return True

            logger.error(f"Download attempt {attempt + 1} failed verification")

        except Exception as e:
            logger.error(f"Error during download attempt {attempt + 1}: {str(e)}")

        if attempt < max_retries - 1:
            logger.info("Cleaning up for retry...")
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            time.sleep(10)

    set_server_status("Error downloading model files")
    return False


def download_model_files_hf(model_repo_id: str, local_model_dir: str):
    try:
        snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)
        return True
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        return False


def check_and_download_model_files():
    """
    Enhanced version of check_and_download_model_files using git-lfs.
    """
    set_server_status("Checking model files...")
    model_config = get_model_config()
    model = model_config.get("model")
    logger.info(f"Checking model files for {model} ...")

    # This is only true if you call it during app startup when running outside the container
    use_local_files = os.environ.get("LOCAL_FILES", "False") == "True"

    # Check if you are running outside the container.. model files will be in a diff location
    if use_local_files:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_dir = os.path.join(current_dir, "..", "..", "model_files", model)
        local_model_dir = os.path.abspath(local_model_dir)
    else:
        local_model_dir = f"/app/model_files/{model}"

    model_config = get_model_config()
    if not model_config:
        error_msg = "Model configuration not found"
        logger.error(error_msg)
        set_server_status(error_msg)
        return False

    model_repo_id = model_config.get("model_repo_id")

    # Always verify existing files even if directory exists
    if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
        # Get expected file size from HF metadata
        expected_sizes = get_expected_file_sizes(model_repo_id)

        logger.info("Verifying existing model files...")
        if verify_download_completion(local_model_dir, expected_sizes):
            logger.info("Existing model files verified successfully")
            set_server_status("ready")
            return True
        else:
            logger.info("Existing files failed verification, re-downloading...")

    # return download_model_files(model_repo_id, local_model_dir)
    return download_model_files_hf(model_repo_id, local_model_dir)
