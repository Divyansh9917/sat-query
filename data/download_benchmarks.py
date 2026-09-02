#!/usr/bin/env python3
"""SatQuery AI — Benchmark Dataset Downloader.

Downloads and extracts the benchmark datasets used for training and evaluation:
  - BigEarthNet-MM (multi-modal Sentinel-1/2)
  - RSVQA (Remote Sensing Visual Question Answering)
  - VRSBench (Visual Remote Sensing Benchmark)
  - CDVQA (Change Detection Visual Question Answering)

Usage:
    python data/download_benchmarks.py --dataset bigearthnet --output data/datasets/
    python data/download_benchmarks.py --all --output data/datasets/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm
from loguru import logger


# ─── Dataset Registry ─────────────────────────────────────────────────────────

DATASETS: Dict[str, Dict[str, str]] = {
    "bigearthnet": {
        "name": "BigEarthNet-MM",
        "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
        "description": (
            "Multi-modal dataset with Sentinel-1 SAR and Sentinel-2 optical "
            "imagery for multi-label land-use classification. ~590k image patches."
        ),
        "size": "~66 GB",
        "format": "tar.gz",
        "sha256": None,  # Add checksum when available
    },
    "rsvqa": {
        "name": "RSVQA",
        "url": "https://zenodo.org/records/6344334/files/RSVQA_LR.zip",
        "description": (
            "Remote Sensing Visual Question Answering dataset with low-resolution "
            "Sentinel-2 images and question-answer pairs."
        ),
        "size": "~2.5 GB",
        "format": "zip",
        "sha256": None,
    },
    "vrsbench": {
        "name": "VRSBench",
        "url": "https://github.com/lx709/VRSBench/archive/refs/heads/main.zip",
        "description": (
            "Visual Remote Sensing Benchmark for image captioning, VQA, "
            "and visual grounding with DIOR-RSVG images."
        ),
        "size": "~500 MB",
        "format": "zip",
        "sha256": None,
    },
    "cdvqa": {
        "name": "CDVQA",
        "url": "https://github.com/YZHJessica/CDVQA/archive/refs/heads/main.zip",
        "description": (
            "Change Detection Visual Question Answering dataset with "
            "bi-temporal image pairs and change-related Q&A."
        ),
        "size": "~1 GB",
        "format": "zip",
        "sha256": None,
    },
}


# ─── Download Utilities ───────────────────────────────────────────────────────

def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 8192,
) -> Path:
    """Download a file with progress bar and optional checksum verification.

    Args:
        url: URL to download.
        dest_path: Local destination path.
        expected_sha256: Optional SHA-256 hash for verification.
        chunk_size: Download chunk size in bytes.

    Returns:
        Path to the downloaded file.
    """
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        if expected_sha256 and verify_checksum(dest_path, expected_sha256):
            logger.info("Checksum verified — skipping download.")
            return dest_path
        elif expected_sha256:
            logger.warning("Checksum mismatch — re-downloading...")
        else:
            logger.info("No checksum available — using existing file.")
            return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {dest_path}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        ncols=80,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Download complete: {dest_path}")

    # Verify checksum
    if expected_sha256:
        if verify_checksum(dest_path, expected_sha256):
            logger.info("✓ Checksum verified.")
        else:
            logger.error("✗ Checksum verification failed!")
            raise RuntimeError(f"Checksum mismatch for {dest_path}")

    return dest_path


def verify_checksum(path: Path, expected: str) -> bool:
    """Verify SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a tar.gz or zip archive.

    Args:
        archive_path: Path to the archive file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the extraction directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    suffix = archive_path.suffix.lower()
    name = archive_path.stem.lower()

    logger.info(f"Extracting: {archive_path} → {extract_dir}")

    if suffix == ".gz" and name.endswith(".tar"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    logger.info(f"Extraction complete: {extract_dir}")
    return extract_dir


# ─── Dataset Download ─────────────────────────────────────────────────────────

def download_dataset(
    dataset_key: str,
    output_dir: Path,
    extract: bool = True,
) -> Path:
    """Download and optionally extract a benchmark dataset.

    Args:
        dataset_key: Key from DATASETS dict.
        output_dir: Base output directory.
        extract: Whether to extract after download.

    Returns:
        Path to the dataset directory.
    """
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset: '{dataset_key}'. "
            f"Available: {list(DATASETS.keys())}"
        )

    info = DATASETS[dataset_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {info['name']}")
    logger.info(f"Size: {info['size']}")
    logger.info(f"Description: {info['description']}")
    logger.info(f"{'='*60}\n")

    # Download
    ext = ".tar.gz" if info["format"] == "tar.gz" else f".{info['format']}"
    archive_name = f"{dataset_key}{ext}"
    archive_path = output_dir / archive_name

    download_file(
        url=info["url"],
        dest_path=archive_path,
        expected_sha256=info.get("sha256"),
    )

    # Extract
    if extract:
        dataset_dir = output_dir / dataset_key
        extract_archive(archive_path, dataset_dir)
        return dataset_dir

    return archive_path


def list_datasets():
    """Print available datasets."""
    print("\n📦 Available Benchmark Datasets:\n")
    for key, info in DATASETS.items():
        print(f"  {key:15s}  {info['name']:20s}  {info['size']:>10s}")
        print(f"  {'':15s}  {info['description'][:70]}...")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI — Download benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python download_benchmarks.py --dataset rsvqa --output data/datasets/\n"
            "  python download_benchmarks.py --all --output data/datasets/\n"
            "  python download_benchmarks.py --list\n"
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/datasets"),
        help="Output directory (default: data/datasets/).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only, don't extract.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets.",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        for key in DATASETS:
            try:
                download_dataset(key, args.output, extract=not args.no_extract)
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                continue
    elif args.dataset:
        download_dataset(args.dataset, args.output, extract=not args.no_extract)
    else:
        parser.print_help()
        print("\n⚠️  Specify --dataset or --all to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
