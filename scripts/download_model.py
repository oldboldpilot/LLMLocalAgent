#!/usr/bin/env python3
"""
Custom script to download Qwen2.5 Coder 7B model using Hugging Face Hub.
Fast, efficient model for local coding assistance.
"""

import os
import sys
from huggingface_hub import HfApi, login
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen2.5-Coder-7B-Instruct"


def check_huggingface_token():
    """Check if HF token is set in environment or prompt for it."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Hugging Face token not found in HF_TOKEN environment variable.")
        print(
            "Please enter your Hugging Face token (won't be stored, only used for this session):"
        )
        token = input().strip()
        if token:
            os.environ["HF_TOKEN"] = token
        else:
            print("Error: No token provided. Cannot download model.")
            sys.exit(1)
    return token


def download_model():
    """Download the model using huggingface_hub with vLLM compatibility."""
    token = check_huggingface_token()

    print(f"Logging into Hugging Face...")
    login(token=token)

    api = HfApi()

    print(f"Model ID: {MODEL_ID}")
    print(f"Local path: {LOCAL_MODEL_PATH}")

    if LOCAL_MODEL_PATH.exists():
        print("Model already exists. Verifying completeness...")
        try:
            api.snapshot_download(
                repo_id=MODEL_ID,
                local_dir=LOCAL_MODEL_PATH,
                resume_download=True,
                token=token,
            )
            print("Model verification/download complete!")
        except Exception as e:
            print(f"Error during download: {e}")
            sys.exit(1)
    else:
        print("Downloading model (this may take a while for ~~14GB download)...")
        try:
            api.snapshot_download(
                repo_id=MODEL_ID,
                local_dir=LOCAL_MODEL_PATH,
                local_dir_use_symlinks=False,
                token=token,
            )
            print(f"Model downloaded successfully to {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error during download: {e}")
            sys.exit(1)

    print(f"\nModel location: {LOCAL_MODEL_PATH}")
    print(f"Model size: {get_size(LOCAL_MODEL_PATH)}")
    return str(LOCAL_MODEL_PATH)


def get_size(path):
    """Get human-readable size of directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024.0:
            return f"{total:.2f} {unit}"
        total /= 1024.0
    return f"{total:.2f} PB"


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2.5 Coder 7B Model Downloader")
    print("=" * 60)
    download_model()
