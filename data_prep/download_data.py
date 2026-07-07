#!/usr/bin/env python3
"""
download_data.py -- Fetch the public base datasets for the face-attribute stage.

Downloads (via the Kaggle API):
  - UTKFace  -> datasets/raw/utkface/    (age + gender, one label pair per face)
  - FER-2013 -> datasets/raw/fer2013/    (facial expression, folder-per-emotion)

--------------------------------------------------------------------------------
ONE-TIME SETUP (Kaggle API token)
--------------------------------------------------------------------------------
  1) Install the client:
         pip install kaggle

  2) Get a token: kaggle.com -> your avatar -> Settings -> "Create New API Token".
     This downloads a file called kaggle.json.

  3) Put kaggle.json where the API looks for it:
         Windows : %USERPROFILE%\\.kaggle\\kaggle.json
         Linux/Mac: ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)

     (Alternatively, set KAGGLE_USERNAME and KAGGLE_KEY as environment variables.)

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python data_prep/download_data.py                # both datasets
  python data_prep/download_data.py --only utkface # just one
  python data_prep/download_data.py --force        # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root = one level up from this file's folder (data_prep/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Kaggle dataset slugs + where they land, relative to the repo root.
DATASETS = {
    "utkface": {
        "slug": "jangedoo/utkface-new",
        "dest": "datasets/raw/utkface",
        "note": "age + gender  (filenames: age_gender_race_date.jpg.chip.jpg)",
    },
    "fer2013": {
        "slug": "msambare/fer2013",
        "dest": "datasets/raw/fer2013",
        "note": "expression   (folders: train/ & test/ -> 7 emotion subfolders)",
    },
    "rafdb": {
        "slug": "shuvoalok/raf-db-dataset",
        "dest": "datasets/raw/rafdb",
        "note": "expression   (real color aligned faces; DATASET/{train,test}/{1-7})",
    },
}


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)
    sys.exit(1)


def _get_api():
    """Import + authenticate the Kaggle client, with friendly error messages."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        _fail(
            "The 'kaggle' package is not installed.\n"
            "  Fix: pip install kaggle\n"
            "  Then re-run this script."
        )

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
        _fail(
            "Kaggle authentication failed -- your API token was not found or is invalid.\n"
            f"  Details: {exc}\n"
            "  Fix: place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json (Windows)\n"
            "       or ~/.kaggle/kaggle.json (Linux/Mac). See the header of this file."
        )
    return api


def _looks_downloaded(dest: Path) -> bool:
    """True if the destination already holds extracted image files."""
    if not dest.exists():
        return False
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if next(dest.rglob(ext), None) is not None:
            return True
    return False


def _count_images(dest: Path) -> int:
    return sum(1 for ext in ("*.jpg", "*.jpeg", "*.png") for _ in dest.rglob(ext))


def download_one(name: str, spec: dict, api, force: bool) -> None:
    dest = (REPO_ROOT / spec["dest"]).resolve()
    print(f"\n=== {name}  ({spec['note']}) ===")
    print(f"    slug: {spec['slug']}")
    print(f"    dest: {dest}")

    if _looks_downloaded(dest) and not force:
        print(f"    SKIP: already present ({_count_images(dest)} images). Use --force to re-download.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print("    downloading + unzipping (this can take a few minutes)...")
    # unzip=True extracts and removes the .zip automatically
    api.dataset_download_files(spec["slug"], path=str(dest), unzip=True, quiet=False)
    print(f"    DONE: {_count_images(dest)} images under {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(DATASETS), help="download just one dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    api = _get_api()
    for name, spec in targets.items():
        download_one(name, spec, api, args.force)

    print("\nAll requested downloads complete.")
    print("Next: build the unified face manifest (age/gender/expression) + splits.")


if __name__ == "__main__":
    main()
