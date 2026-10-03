#!/usr/bin/env python3
"""Dataset Downloader for GTA Benchmark.

Downloads and extracts:
- GTA-Workflow (v2) dataset to opencompass/data/gta_dataset_v2
- GTA-Atomic (v1) dataset to opencompass/data/gta_dataset
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

URLS = {
    "workflow": "https://github.com/open-compass/GTA/releases/download/v0.2.0/gta_workflow_dataset.zip",
    "atomic": "https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip",
}

TARGET_DIRS = {
    "workflow": REPO_ROOT / "opencompass" / "data" / "gta_dataset_v2",
    "atomic": REPO_ROOT / "opencompass" / "data" / "gta_dataset",
}

def download_file(url: str, dest_path: Path):
    print(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = block_num * block_size * 100 / total_size
            downloaded_mb = (block_num * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\nDownload complete.")

def extract_zip(zip_path: Path, target_dir: Path):
    print(f"Extracting {zip_path.name}...")
    temp_dir = target_dir.parent / f"_tmp_extract_{zip_path.stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    extracted_items = [p for p in temp_dir.iterdir() if p.name not in ("__MACOSX", ".DS_Store")]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        src_dir = extracted_items[0]
    else:
        src_dir = temp_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dest = target_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    if src_dir != temp_dir and src_dir.name != target_dir.name:
        alias_link = target_dir.parent / src_dir.name
        if not alias_link.exists():
            try:
                alias_link.symlink_to(target_dir.name)
            except Exception:
                pass

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Extraction complete: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download GTA datasets.")
    parser.add_argument(
        "--dataset",
        choices=["workflow", "atomic", "all"],
        default="workflow",
        help="Which dataset to download (workflow / atomic / all). Default: workflow",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / "opencompass" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = ["workflow", "atomic"] if args.dataset == "all" else [args.dataset]

    for key in targets:
        url = URLS[key]
        dest_zip = data_dir / f"{key}_dataset.zip"
        target_dir = TARGET_DIRS[key]

        print(f"\n=== Preparing {key.upper()} Dataset ===")
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"Directory {target_dir} already exists and is not empty. Skipping download.")
            continue

        try:
            download_file(url, dest_zip)
            extract_zip(dest_zip, target_dir)
            if dest_zip.exists():
                dest_zip.unlink()
            print(f"[SUCCESS] {key.upper()} dataset ready at {target_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download or unpack {key}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
