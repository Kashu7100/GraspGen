import argparse
from huggingface_hub import snapshot_download
import os

def download_checkpoints():
    """Download checkpoints from Hugging Face: https://huggingface.co/adithyamurali/GraspGenModels"""
    repo_id = "adithyamurali/GraspGenModels"
    local_dir = os.path.join(os.getcwd(), "models")
    
    print(f"Downloading GraspGen models from {repo_id} to {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print("Download complete!")
    except Exception as e:
        print(f"Failed to download models: {e}")

def main():
    parser = argparse.ArgumentParser(description="GraspGen CLI tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 'download' command
    download_parser = subparsers.add_parser('download', help='Download model checkpoints')

    args = parser.parse_args()

    if args.command == 'download':
        download_checkpoints()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
