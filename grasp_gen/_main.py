from __future__ import annotations
from typing import Optional

import argparse
from huggingface_hub import snapshot_download
import os


def download_checkpoints(save_dir: Optional[str] = None):
    """Download checkpoints from Hugging Face: https://huggingface.co/adithyamurali/GraspGenModels"""
    repo_id = "adithyamurali/GraspGenModels"
    local_dir = save_dir or os.path.join(os.getcwd(), "gg_models")

    print(f"Downloading GraspGen models from {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False
        )
        print(f"Download complete!")
        print(f"\nPlease run:\nexport GRASPGEN_CHECKPOINT_DIR={local_dir}/checkpoints")
    except Exception as e:
        print(f"Failed to download models: {e}")


def main():
    parser = argparse.ArgumentParser(description="GraspGen CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'download' command
    download_parser = subparsers.add_parser(
        "download", help="Download model checkpoints"
    )
    download_parser.add_argument(
        "--save-dir", type=str, help="Directory to save models"
    )

    args = parser.parse_args()

    if args.command == "download":
        download_checkpoints(args.save_dir) if args.save_dir else download_checkpoints()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
