#!/usr/bin/env python3
"""
upload_to_hf.py -- push UTKFace (as one Store-zip) + the manifest to a private HF dataset repo.

Modal downloads these from HF at train time, so we ship ONE big archive (fast, reliable)
instead of 23k tiny files. The zip keeps the repo-relative paths
(datasets/raw/utkface/UTKFace/...) so it extracts straight onto the Volume and the
manifest's paths resolve unchanged.

Usage:  python data_prep/upload_to_hf.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
UTK_DIR = REPO_ROOT / "datasets" / "raw" / "utkface" / "UTKFace"
MANIFEST = REPO_ROOT / "datasets" / "processed" / "faces_manifest.csv"
ZIP_PATH = REPO_ROOT / "datasets" / "utkface.zip"
HF_REPO_ID = "AwsHanaqtah/utkface-gender-age"


def make_zip() -> None:
    if ZIP_PATH.exists():
        print(f"zip already exists: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")
        return
    files = [p for p in UTK_DIR.iterdir() if p.is_file()]
    print(f"zipping {len(files)} images (Store / no compression)...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(files):
            zf.write(p, p.relative_to(REPO_ROOT).as_posix())
            if i % 4000 == 0:
                print(f"  {i}/{len(files)}")
    print(f"wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")


def upload() -> None:
    create_repo(HF_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api = HfApi()
    print("uploading utkface.zip ...")
    api.upload_file(path_or_fileobj=str(ZIP_PATH), path_in_repo="utkface.zip",
                    repo_id=HF_REPO_ID, repo_type="dataset")
    print("uploading faces_manifest.csv ...")
    api.upload_file(path_or_fileobj=str(MANIFEST), path_in_repo="faces_manifest.csv",
                    repo_id=HF_REPO_ID, repo_type="dataset")
    print(f"done -> https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    make_zip()
    upload()
