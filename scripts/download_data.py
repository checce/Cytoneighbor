#!/usr/bin/env python3
"""Download the study's AnnData inputs from the public dataset folder."""
import argparse
import csv
import hashlib
from pathlib import Path
import tempfile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FOLDER = "https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE"
FILE_IDS = {
    "Human immune.h5ad": "1s-nOAlrCboVoDQAJLyosBWLK5CqY3NOb",
    "Pancreas.h5ad": "1sQZ3gTouSTuuSRM8OnIDKdMxTVP63s3l",
    "Diabetic Kidney Disease.h5ad": "1C4gCWUUtmQaGibCGA7r8x9HJn854nW8W",
    "Lung_atlas_public.h5ad": "1EBGy-iHVdbn2BVoIKFAWq1PsyHV7tKsl",
    "Mouse_Kidney_mod1.h5ad": "1YfYDIvK_P8FVv5BkZEKCgODAkW_6TQKg",
    "8d2e51ab-0edd-46c8-bdac-558de319d6c2_sample_id_7_donor_id_6.h5ad": "1pG-E9uqRfq0BFXXbeN0_whZP-OOBHBCi",
    "Zebrafish embryonic cells.h5ad": None,
}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name, expected):
    destination = DATA_DIR / name
    if destination.is_file():
        print(f"Exists: {name}. Use --verify to compare it with the study input.")
        return True
    if FILE_IDS[name] is None:
        print(f"No public file is linked for {name}. See data/README.md.")
        return False
    try:
        import gdown
    except ImportError:
        print("Install gdown with python -m pip install gdown, or download manually.")
        return False

    # Check the input before putting it at the path consumed by experiments.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=DATA_DIR) as temp:
        partial = Path(temp) / name
        try:
            gdown.download(id=FILE_IDS[name], output=str(partial))
            if (not partial.is_file()
                    or partial.stat().st_size != int(expected["Bytes"])
                    or fingerprint(partial) != expected["SHA256"]):
                print(f"Download did not match the study checksum: {name}")
                return False
            if destination.exists():
                print(f"A file appeared at {destination} during download. Verify it before use.")
                return False
            partial.rename(destination)
        except Exception as exc:
            print(f"Could not download {name}: {exc}")
            return False
    print(f"Downloaded and verified {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Show local and public availability")
    mode.add_argument("--verify", action="store_true", help="Verify local files by SHA-256")
    parser.add_argument("--file", choices=FILE_IDS, help="Use a single exact filename")
    args = parser.parse_args()
    names = [args.file] if args.file else list(FILE_IDS)
    print(f"Dataset folder: {FOLDER}\n")
    if args.list:
        for name in names:
            local = "present" if (DATA_DIR / name).is_file() else "missing"
            public = "linked" if FILE_IDS[name] else "not linked"
            print(f"{name}: {local} locally, {public} on Drive")
        return 0

    with (DATA_DIR / "input-checksums.csv").open(encoding="utf-8-sig", newline="") as stream:
        checksums = {row["Filename"]: row for row in csv.DictReader(stream)}
    if args.verify:
        ok = True
        for name in names:
            path = DATA_DIR / name
            expected = checksums[name]
            matches = (path.is_file() and path.stat().st_size == int(expected["Bytes"])
                       and fingerprint(path) == expected["SHA256"])
            print(f"{'OK' if matches else 'Missing or different'}: {name}")
            ok &= matches
        return 0 if ok else 1

    # The default action downloads the currently linked files.
    if args.file is None:
        missing_links = [name for name in names if FILE_IDS[name] is None]
        if missing_links:
            print("No public download link for: " + ", ".join(missing_links))
        names = [name for name in names if FILE_IDS[name]]
    results = [download(name, checksums[name]) for name in names]
    if not all(results):
        print(f"See data/README.md for manual download instructions and expected checksums.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
