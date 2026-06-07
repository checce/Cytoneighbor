#!/usr/bin/env python3
"""
Download Cytoneighbor datasets from Google Drive.

This script attempts to download the required .h5ad files using gdown.
If gdown is unavailable or a file cannot be downloaded automatically,
it prints manual download instructions.

Usage:
    python scripts/download_data.py          # download all datasets
    python scripts/download_data.py --list   # list datasets only
"""

import os
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry:  { filename: google_drive_file_id }
# Replace the placeholder IDs with actual Google Drive file IDs.
# To obtain a file ID, right-click the file in Google Drive and select
# "Get link" -> the ID is the string after "/d/" and before "/view".
# ---------------------------------------------------------------------------
DATASETS = {
    "Human immune.h5ad":             "PASTE_FILE_ID_HERE",
    "Pancreas.h5ad":                 "PASTE_FILE_ID_HERE",
    "Diabetic Kidney Disease.h5ad":  "PASTE_FILE_ID_HERE",
    "Lung_atlas_public.h5ad":        "PASTE_FILE_ID_HERE",
    "Mouse_Kidney_mod1.h5ad":        "PASTE_FILE_ID_HERE",
    "Zebrafish embryonic cells.h5ad":"PASTE_FILE_ID_HERE",
}

GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def list_datasets():
    """Print the dataset inventory."""
    print("\nCytoneighbor Datasets")
    print("=" * 60)
    print(f"Google Drive: {GOOGLE_DRIVE_FOLDER}\n")
    for fname, fid in DATASETS.items():
        status = "READY" if (DATA_DIR / fname).exists() else "MISSING"
        print(f"  [{status:7s}] {fname}")
    print()


def download_with_gdown(fname, file_id, quiet=False):
    """Download a single file using gdown."""
    try:
        import gdown
    except ImportError:
        print("  [!] gdown not installed. Run: pip install gdown")
        return False

    dest = DATA_DIR / fname
    if dest.exists():
        print(f"  [SKIP] {fname} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  [->] Downloading {fname} ...")
    try:
        gdown.download(url, str(dest), quiet=quiet)
        if dest.exists():
            print(f"  [OK]  {fname} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
    except Exception as e:
        print(f"  [FAIL] {fname}: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Download Cytoneighbor datasets")
    parser.add_argument("--list", action="store_true", help="List datasets only")
    parser.add_argument("--file", type=str, help="Download a specific file by name")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets()
        return

    # Check if any file IDs are still placeholders
    all_placeholders = all(fid == "PASTE_FILE_ID_HERE" for fid in DATASETS.values())
    if all_placeholders:
        print("=" * 60)
        print("  Google Drive file IDs have not been configured yet.")
        print("  Please edit scripts/download_data.py and replace")
        print("  'PASTE_FILE_ID_HERE' with actual Google Drive file IDs.")
        print("=" * 60)
        print()
        print(f"  Folder: {GOOGLE_DRIVE_FOLDER}")
        print()
        print("  To obtain file IDs:")
        print("    1. Open the Google Drive folder")
        print("    2. Right-click each file -> 'Get link' -> 'Copy link'")
        print("    3. The ID is between '/d/' and '/view' in the URL")
        print()
        print("  Alternatively, download files manually and place them in:")
        print(f"    {DATA_DIR.resolve()}")
        print()
        list_datasets()
        return

    if args.file:
        if args.file not in DATASETS:
            print(f"Unknown dataset: {args.file}")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        targets = {args.file: DATASETS[args.file]}
    else:
        targets = DATASETS

    print("\nDownloading Cytoneighbor datasets ...")
    print(f"Target directory: {DATA_DIR.resolve()}\n")

    success = 0
    for fname, fid in targets.items():
        if download_with_gdown(fname, fid):
            success += 1

    print(f"\nDone: {success}/{len(targets)} datasets ready.\n")

    # Remind about remaining files
    missing = [f for f in targets if not (DATA_DIR / f).exists()]
    if missing:
        print("The following files could not be downloaded automatically:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nPlease download them manually from:\n  {GOOGLE_DRIVE_FOLDER}")
        print(f"and place them in:\n  {DATA_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
