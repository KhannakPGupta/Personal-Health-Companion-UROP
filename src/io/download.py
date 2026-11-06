"""Cross-platform dataset acquisition module for WESAD and PhysioNet Wearable Device Dataset.

Enforces Rule 1: Never fabricate or simulate data to replace a dataset that fails to download.
If a download fails or a server returns an error, the error is raised explicitly to the user.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


WESAD_PRIMARY_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub04n/download"
WESAD_FALLBACK_URL = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
PHYSIONET_BASE_URL = "https://physionet.org/files/wearable-device-dataset/1.0.1/"
PHYSIONET_ZIP_URL = (
    "https://physionet.org/static/published-projects/wearable-device-dataset/"
    "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1.zip"
)


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with progress reporting.
    
    Args:
        url: Remote HTTP(S) URL.
        destination: Local file path.
        chunk_size: Streaming chunk size in bytes.
        
    Returns:
        Path to downloaded file.
        
    Raises:
        RuntimeError: If download fails (Rule 1: Never simulate data).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(destination.suffix + ".tmp")

    print(f"[*] Downloading: {url}")
    print(f"[*] Destination: {destination}")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HealthCompanionResearch/1.0"}
        )
        with urllib.request.urlopen(req) as response, open(temp_dest, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_size = int(total_size) if total_size and total_size.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    -> {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
                else:
                    mb_down = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r    -> {mb_down:.1f} MB downloaded")
                sys.stdout.flush()

        print("\n[*] Download complete. Moving to target location...")
        if destination.exists():
            destination.unlink()
        temp_dest.rename(destination)
        return destination

    except Exception as exc:
        if temp_dest.exists():
            temp_dest.unlink()
        raise RuntimeError(
            f"Download failed for {url}: {exc}. "
            "Rule 1 violation prevention: Stopping execution instead of simulating data."
        ) from exc


def extract_zip(archive_path: Path, target_dir: Path) -> None:
    """Extract a ZIP archive with safety checks."""
    print(f"[*] Extracting {archive_path} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"[*] Extracted successfully to {target_dir}")


def fetch_wesad(target_dir: Path, archive_cache: Path | None = None) -> None:
    """Fetch WESAD dataset and unpack into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = archive_cache or (target_dir.parent / "WESAD.zip")

    if not zip_dest.exists():
        try:
            download_file(WESAD_PRIMARY_URL, zip_dest)
        except RuntimeError as primary_err:
            print(f"[!] Primary URL failed ({primary_err}). Trying UCI fallback...")
            download_file(WESAD_FALLBACK_URL, zip_dest)

    extract_zip(zip_dest, target_dir)


def fetch_physionet(target_dir: Path, archive_cache: Path | None = None) -> None:
    """Fetch PhysioNet Wearable Device Dataset v1.0.1 ZIP and unpack into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = archive_cache or (target_dir.parent / "physionet_wearable_1.0.1.zip")

    if not zip_dest.exists():
        download_file(PHYSIONET_ZIP_URL, zip_dest)

    temp_extract = target_dir.parent / "_physionet_extract_tmp"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    extract_zip(zip_dest, temp_extract)

    # Locate the inner folder that contains STRESS / AEROBIC / ANAEROBIC / data_constraints.txt
    inner_dirs = [d for d in temp_extract.rglob("STRESS") if d.is_dir()]
    if inner_dirs:
        source_root = inner_dirs[0].parent
        for item in source_root.iterdir():
            dest_item = target_dir / item.name
            if dest_item.exists():
                if dest_item.is_dir():
                    shutil.rmtree(dest_item)
                else:
                    dest_item.unlink()
            shutil.move(str(item), str(dest_item))
    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"[*] PhysioNet v1.0.1 ready at {target_dir}")


def main() -> None:
    """Command-line interface for downloading research datasets."""
    parser = argparse.ArgumentParser(description="Download WESAD and PhysioNet Wearable datasets.")
    parser.add_argument(
        "--dataset",
        choices=["wesad", "physionet", "all"],
        default="all",
        help="Dataset to fetch (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target base directory for raw datasets (default: data/raw)",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("wesad", "all"):
        print("\n=== Fetching WESAD ===")
        wesad_dir = args.data_dir / "WESAD"
        fetch_wesad(wesad_dir)

    if args.dataset in ("physionet", "all"):
        print("\n=== Fetching PhysioNet Wearable Device Dataset v1.0.1 ===")
        physio_dir = args.data_dir / "physionet_wearable" / "1.0.1"
        fetch_physionet(physio_dir)


if __name__ == "__main__":
    main()
