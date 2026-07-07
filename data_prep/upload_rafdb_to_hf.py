#!/usr/bin/env python3
"""
upload_rafdb_to_hf.py -- push RAF-DB (as one Store-zip) to a private HF dataset repo.

Mirrors upload_to_hf.py, but RAF-DB needs no manifest -- labels are the folder
names (DATASET/{train,test}/{1-7}), which ExpressionDataset reads directly. The
zip keeps the repo-relative paths so Modal extracts it straight onto the Volume.

Usage:  python data_prep/upload_rafdb_to_hf.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
RAFDB_DIR = REPO_ROOT / "datasets" / "raw" / "rafdb" / "DATASET"
ZIP_PATH = REPO_ROOT / "datasets" / "rafdb.zip"
HF_REPO_ID = "AwsHanaqtah/rafdb-expression"


def make_zip() -> None:
    if ZIP_PATH.exists():
        print(f"zip already exists: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")
        return
    files = [p for p in RAFDB_DIR.rglob("*.jpg")]
    print(f"zipping {len(files)} images (Store / no compression)...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(files):
            zf.write(p, p.relative_to(REPO_ROOT).as_posix())
            if i % 3000 == 0:
                print(f"  {i}/{len(files)}")
    print(f"wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")


def upload() -> None:
    create_repo(HF_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api = HfApi()
    print("uploading rafdb.zip ...")
    api.upload_file(path_or_fileobj=str(ZIP_PATH), path_in_repo="rafdb.zip",
                    repo_id=HF_REPO_ID, repo_type="dataset")
    print(f"done -> https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    make_zip()
    upload()
