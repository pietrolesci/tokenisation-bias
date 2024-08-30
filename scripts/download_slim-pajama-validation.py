import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

LOCAL_DIR = Path("data/slim-pajama-validation")

if __name__ == "__main__":
    # Download
    snapshot_download(
        "cerebras/SlimPajama-627B",
        revision="refs/convert/parquet",
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
        allow_patterns=["default/partial-validation/*"],
    )

    # Move files under parent
    path = LOCAL_DIR / "default" / "partial-validation"
    for filepath in path.rglob("*.parquet"):
        shutil.move(filepath, str(LOCAL_DIR))

    # Remove useless folders
    shutil.rmtree(LOCAL_DIR / ".cache", ignore_errors=True)
    shutil.rmtree(LOCAL_DIR / "default", ignore_errors=True)
