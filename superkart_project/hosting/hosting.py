import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER", "JefferyMendis")

# Hugging Face Space name (use lowercase + hyphens)
SPACE_NAME = os.getenv("SPACE_NAME", "superkart-sales-app")

# Local folder containing deployment files (Dockerfile, app.py, requirements.txt)
LOCAL_FOLDER = os.getenv(
    "LOCAL_FOLDER",
    "superkart_project/deployment"
)

REPO_ID = f"{HF_USER}/{SPACE_NAME}"
REPO_TYPE = "space"

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable not set. "
        "Please provide a Hugging Face token with write permissions."
    )

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------
# Space configuration
# ---------------------------------------------------------
# Using Docker SDK (requires Dockerfile in LOCAL_FOLDER)
SPACE_SDK = "docker"

# ---------------------------------------------------------
# Create Space if it does not exist
# ---------------------------------------------------------
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"✅ Hugging Face Space '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    try:
        print(f"🚀 Creating Hugging Face Space '{REPO_ID}' with SDK='{SPACE_SDK}'...")
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            space_sdk=SPACE_SDK
        )
        print("✅ Space created successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to create Space: {e}") from e

# ---------------------------------------------------------
# Upload deployment files to the Space
# ---------------------------------------------------------
try:
    print(f"📦 Uploading deployment files from '{LOCAL_FOLDER}' to Space '{REPO_ID}'...")
    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=""  # upload to repo root
    )
    print("✅ Upload completed successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to upload files to Space: {e}") from e
