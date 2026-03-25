"""Upload the converted HuggingFace model folder to the Hub.

Environment variables:
  REPO_ID        - HuggingFace repository ID (required)
  HF_TOKEN       - HuggingFace API token with write access (required)
  COMMIT_MESSAGE - Commit message for the upload (optional)
"""

import os
from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
commit_message = os.environ.get(
    "COMMIT_MESSAGE", "Deploy EchoSelf NanEcho model"
)
token = os.environ.get("HF_TOKEN")
if not token:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "A HuggingFace token with write access is required to upload."
    )

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="hf-model/",
    repo_id=repo_id,
    repo_type="model",
    commit_message=commit_message,
)
