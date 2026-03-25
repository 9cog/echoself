"""Upload the converted HuggingFace model folder to the Hub.

Environment variables:
  REPO_ID        - HuggingFace repository ID (required)
  COMMIT_MESSAGE - Commit message for the upload (optional)
"""

import os
from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
commit_message = os.environ.get(
    "COMMIT_MESSAGE", "Deploy EchoSelf NanEcho model"
)

api = HfApi()
api.upload_folder(
    folder_path="hf-model/",
    repo_id=repo_id,
    repo_type="model",
    commit_message=commit_message,
)
