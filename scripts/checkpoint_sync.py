#!/usr/bin/env python3
"""
Checkpoint sync utility for Hugging Face Hub.

Primary storage: Hugging Face Hub (no size limits, ML-optimized)
Backup storage: Git LFS (configured in .gitattributes)

Usage:
    # Upload checkpoint
    python scripts/checkpoint_sync.py upload --checkpoint out-nanecho/ckpt.pt

    # Download checkpoint to resume training
    python scripts/checkpoint_sync.py download --checkpoint out-nanecho/ckpt.pt

    # List available checkpoints
    python scripts/checkpoint_sync.py list
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")


# Configuration - update these for your setup
DEFAULT_REPO_ID = os.environ.get("HF_CHECKPOINT_REPO", "")
DEFAULT_REPO_TYPE = "model"


def get_repo_id():
    """Get HF repo ID from env or prompt user."""
    repo_id = DEFAULT_REPO_ID
    if not repo_id:
        repo_id = os.environ.get("HF_CHECKPOINT_REPO")
    if not repo_id:
        print("Error: Set HF_CHECKPOINT_REPO environment variable or update DEFAULT_REPO_ID in this script")
        print("Example: export HF_CHECKPOINT_REPO='your-username/echoself-checkpoints'")
        sys.exit(1)
    return repo_id


def upload_checkpoint(checkpoint_path: str, repo_id: str = None):
    """Upload checkpoint to Hugging Face Hub."""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return False

    repo_id = repo_id or get_repo_id()
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False

    # Create versioned path in repo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = checkpoint_path.name

    api = HfApi()

    try:
        # Upload as latest
        print(f"Uploading {checkpoint_path} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"checkpoints/{filename}",
            repo_id=repo_id,
            repo_type=DEFAULT_REPO_TYPE,
        )

        # Also save versioned copy
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"checkpoints/archive/{filename.replace('.pt', '')}_{timestamp}.pt",
            repo_id=repo_id,
            repo_type=DEFAULT_REPO_TYPE,
        )

        print(f"✓ Uploaded to: https://huggingface.co/{repo_id}")
        print(f"  - checkpoints/{filename} (latest)")
        print(f"  - checkpoints/archive/{filename.replace('.pt', '')}_{timestamp}.pt (versioned)")
        return True

    except Exception as e:
        print(f"Error uploading: {e}")
        print("Make sure you're logged in: huggingface-cli login")
        return False


def download_checkpoint(checkpoint_path: str, repo_id: str = None, version: str = None):
    """Download checkpoint from Hugging Face Hub."""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return False

    repo_id = repo_id or get_repo_id()
    checkpoint_path = Path(checkpoint_path)
    filename = checkpoint_path.name

    # Determine which file to download
    if version:
        repo_path = f"checkpoints/archive/{version}"
    else:
        repo_path = f"checkpoints/{filename}"

    try:
        print(f"Downloading {repo_path} from {repo_id}...")

        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=DEFAULT_REPO_TYPE,
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False,
        )

        # Move to expected location if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != checkpoint_path:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            downloaded_path.rename(checkpoint_path)

        print(f"✓ Downloaded to: {checkpoint_path}")
        return True

    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def list_checkpoints(repo_id: str = None):
    """List available checkpoints on Hugging Face Hub."""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return

    repo_id = repo_id or get_repo_id()

    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE)

        checkpoints = [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")]

        if not checkpoints:
            print(f"No checkpoints found in {repo_id}")
            return

        print(f"Checkpoints in {repo_id}:")
        print("\nLatest:")
        for f in checkpoints:
            if "/archive/" not in f:
                print(f"  - {f}")

        archived = [f for f in checkpoints if "/archive/" in f]
        if archived:
            print("\nArchived versions:")
            for f in sorted(archived, reverse=True)[:10]:  # Show last 10
                print(f"  - {f}")
            if len(archived) > 10:
                print(f"  ... and {len(archived) - 10} more")

    except Exception as e:
        print(f"Error listing checkpoints: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync checkpoints with Hugging Face Hub")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload checkpoint to HF Hub")
    upload_parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint file")
    upload_parser.add_argument("--repo", "-r", help="HF repo ID (default: $HF_CHECKPOINT_REPO)")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download checkpoint from HF Hub")
    download_parser.add_argument("--checkpoint", "-c", required=True, help="Local path to save checkpoint")
    download_parser.add_argument("--repo", "-r", help="HF repo ID (default: $HF_CHECKPOINT_REPO)")
    download_parser.add_argument("--version", "-v", help="Specific version to download (from archive)")

    # List command
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument("--repo", "-r", help="HF repo ID (default: $HF_CHECKPOINT_REPO)")

    args = parser.parse_args()

    if args.command == "upload":
        success = upload_checkpoint(args.checkpoint, args.repo)
        sys.exit(0 if success else 1)
    elif args.command == "download":
        success = download_checkpoint(args.checkpoint, args.repo, args.version)
        sys.exit(0 if success else 1)
    elif args.command == "list":
        list_checkpoints(args.repo)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
