"""Upload the converted HuggingFace model folder to the Hub.

Environment variables:
  REPO_ID        - HuggingFace repository ID (required); may be a full URL,
                   e.g. https://huggingface.co/namespace/repo — it will be
                   normalised to the 'namespace/repo_name' form automatically.
  HF_TOKEN       - HuggingFace API token with write access (required)
  COMMIT_MESSAGE - Commit message for the upload (optional)
"""

import os
import re
from huggingface_hub import HfApi


def _normalise_repo_id(raw: str) -> str:
    """Return a valid HuggingFace repo ID from *raw*.

    Handles full Hub URLs such as:
      https://huggingface.co/namespace/repo-name
      http://huggingface.co/namespace/repo-name
    as well as plain 'namespace/repo-name' values.
    """
    raw = raw.strip().rstrip("/")
    # Strip scheme + host if a URL was provided
    match = re.match(
        r"^https?://[^/]+/(.+)$",
        raw,
        re.IGNORECASE,
    )
    if match:
        raw = match.group(1)
    return raw


repo_id = _normalise_repo_id(os.environ["REPO_ID"])
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
