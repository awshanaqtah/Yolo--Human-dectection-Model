#!/usr/bin/env python3
"""
build_manifest.py -- Unify the public face datasets into ONE labeled manifest.

Sources (already downloaded by download_data.py):
  - UTKFace  datasets/raw/utkface/UTKFace/   -> age + gender  (from filename)
  - FER-2013 datasets/raw/fer2013/{train,test}/<emotion>/ -> expression (from folder)

Output:
  - datasets/processed/faces_manifest.csv   one row per image, columns:
        image_path, source, split, gender, age, expression
    Missing labels are left blank (each source only fills what it has).
  - datasets/processed/summary.txt          human-readable counts / balance report

Design notes:
  * We use ONLY utkface/UTKFace/ (23,708 aligned faces). The Kaggle mirror also
    ships crop_part1/ and a duplicate utkface_aligned_cropped/ -- both ignored.
  * UTKFace is split 90/5/5 train/val/test, STRATIFIED by gender x age-decade so
    both gender and age stay balanced across the splits. (The age-decade bin is
    used ONLY for balancing here; it is never written to the manifest -- age is
    stored as the exact integer.)
  * FER-2013 keeps its official test split; its train is split 90/10 into
    train/val, STRATIFIED by emotion (important -- 'disgust' has only 436 images).
  * Exact-duplicate images (same bytes) are detected via SHA1 and dropped
    (kept once) so identical files can't leak across splits.
  * Standard library only -- no pandas/sklearn needed.

Usage:
  python data_prep/build_manifest.py
  python data_prep/build_manifest.py --no-dedup      # skip the hash pass (faster)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "datasets" / "raw"
UTK_DIR = RAW / "utkface" / "UTKFace"
FER_DIR = RAW / "fer2013"
OUT_DIR = REPO_ROOT / "datasets" / "processed"
MANIFEST = OUT_DIR / "faces_manifest.csv"
SUMMARY = OUT_DIR / "summary.txt"

# ----------------------------------------------------------------------------- config
SEED = 42
UTK_SPLIT = {"train": 0.90, "val": 0.05, "test": 0.05}
FER_VAL_FRAC = 0.10  # carve val out of FER's official train

# UTKFace encodes gender as a digit in the filename; per the UTKFace spec 0=male, 1=female.
# This map DECODES that digit into a word (digit is the key because that's what we read).
GENDER_DECODE = {0: "male", 1: "female"}

FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

FIELDS = ["image_path", "source", "split", "gender", "age", "expression"]


def rel(path: Path) -> str:
    """Store paths relative to the repo root (portable to Modal's mounted volume)."""
    return path.relative_to(REPO_ROOT).as_posix()


def age_decade(age: int) -> int:
    """Coarse age bin (0..9) used ONLY to balance the split -- not stored."""
    return min(age // 10, 9)


# ----------------------------------------------------------------------------- UTKFace
def parse_utkface() -> tuple[list[dict], int]:
    """Return (rows, num_malformed). Each row has gender + age filled."""
    rows: list[dict] = []
    malformed = 0
    for p in UTK_DIR.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        # filename: age_gender_race_datetime.jpg.chip.jpg  -> take the first token
        parts = p.name.split(".")[0].split("_")
        try:
            age = int(parts[0])
            gender = int(parts[1])
            if not (0 <= age <= 116) or gender not in GENDER_DECODE:
                raise ValueError
        except (ValueError, IndexError):
            malformed += 1
            continue
        rows.append({
            "image_path": rel(p),
            "source": "utkface",
            "split": "",  # assigned later
            "gender": GENDER_DECODE[gender],
            "age": age,
            "expression": "",
        })
    return rows, malformed


# ----------------------------------------------------------------------------- FER-2013
def parse_fer() -> list[dict]:
    rows: list[dict] = []
    for official in ("train", "test"):
        for emotion in FER_EMOTIONS:
            folder = FER_DIR / official / emotion
            if not folder.is_dir():
                continue
            for p in folder.iterdir():
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                rows.append({
                    "image_path": rel(p),
                    "source": "fer2013",
                    # official test stays test; official train assigned train/val later
                    "split": "test" if official == "test" else "",
                    "gender": "",
                    "age": "",
                    "expression": emotion,
                })
    return rows


# ----------------------------------------------------------------------------- splitting
def assign_stratified(rows: list[dict], key_fn, fracs: dict[str, float], seed: int) -> None:
    """Assign row['split'] in place, keeping each stratum's proportion in every split."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    rng = random.Random(seed)
    split_names = list(fracs)
    for items in groups.values():
        rng.shuffle(items)
        n = len(items)
        idx = 0
        for name in split_names[:-1]:            # all but the last split get a fixed cut
            take = int(round(fracs[name] * n))
            for r in items[idx:idx + take]:
                r["split"] = name
            idx += take
        for r in items[idx:]:                    # last split takes the remainder
            r["split"] = split_names[-1]


# ----------------------------------------------------------------------------- dedup
def drop_exact_duplicates(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    dupes = 0
    for r in tqdm(rows, desc="hashing (dedup)"):
        h = hashlib.sha1((REPO_ROOT / r["image_path"]).read_bytes()).hexdigest()
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        kept.append(r)
    return kept, dupes


# ----------------------------------------------------------------------------- report
def build_summary(rows: list[dict], utk_malformed: int, dupes: int) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 64)
    add("FACE MANIFEST SUMMARY")
    add("=" * 64)
    add(f"total rows: {len(rows)}")
    add(f"utkface malformed filenames skipped: {utk_malformed}")
    add(f"exact-duplicate images dropped:      {dupes}")
    add("")

    by_source = Counter(r["source"] for r in rows)
    add("-- rows per source --")
    for k, v in by_source.items():
        add(f"  {k:10s} {v}")
    add("")

    add("-- split x source --")
    sx = Counter((r["source"], r["split"]) for r in rows)
    for src in by_source:
        parts = [f"{sp}={sx[(src, sp)]}" for sp in ("train", "val", "test")]
        add(f"  {src:10s} " + "  ".join(parts))
    add("")

    utk = [r for r in rows if r["source"] == "utkface"]
    if utk:
        add("-- UTKFace: gender --")
        for k, v in Counter(r["gender"] for r in utk).most_common():
            add(f"  {k:8s} {v}")
        ages = [r["age"] for r in utk]
        add(f"-- UTKFace: age  range {min(ages)}..{max(ages)}  mean {sum(ages)/len(ages):.1f} --")
        add("")

    fer = [r for r in rows if r["source"] == "fer2013"]
    if fer:
        add("-- FER-2013: expression (imbalanced!) --")
        for k, v in Counter(r["expression"] for r in fer).most_common():
            add(f"  {k:10s} {v}")
        add("")

    add("wrote: " + rel(MANIFEST))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dedup", action="store_true", help="skip the SHA1 duplicate pass")
    args = ap.parse_args()

    if not UTK_DIR.is_dir():
        raise SystemExit(f"UTKFace not found at {UTK_DIR} -- run download_data.py first")
    if not FER_DIR.is_dir():
        raise SystemExit(f"FER-2013 not found at {FER_DIR} -- run download_data.py first")

    print("parsing UTKFace...")
    utk_rows, utk_malformed = parse_utkface()
    print("parsing FER-2013...")
    fer_rows = parse_fer()

    rows = utk_rows + fer_rows

    dupes = 0
    if not args.no_dedup:
        rows, dupes = drop_exact_duplicates(rows)

    # assign splits AFTER dedup so proportions are exact
    utk_rows = [r for r in rows if r["source"] == "utkface"]
    fer_train = [r for r in rows if r["source"] == "fer2013" and r["split"] != "test"]

    assign_stratified(utk_rows, lambda r: (r["gender"], age_decade(r["age"])), UTK_SPLIT, SEED)
    assign_stratified(
        fer_train, lambda r: r["expression"],
        {"train": 1 - FER_VAL_FRAC, "val": FER_VAL_FRAC}, SEED,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows, utk_malformed, dupes)
    SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
