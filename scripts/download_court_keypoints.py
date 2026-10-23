"""
Download the court keypoints dataset from Roboflow.

Dataset: reloc2 (basketball court keypoints)
  - 1,468 images with 18 court keypoints annotated
  - Used to train the learned homography keypoint detector
  - Source: https://universe.roboflow.com/relocalisation/reloc2

Usage:
    python scripts/download_court_keypoints.py --api-key YOUR_KEY
    python scripts/download_court_keypoints.py  # prompts for key interactively
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

# Load .env from project root (silently ignored if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to env vars / CLI arg


ROBOFLOW_WORKSPACE = "relocalisation"
ROBOFLOW_PROJECT   = "reloc2"
ROBOFLOW_VERSION   = 2          # latest stable version
EXPORT_FORMAT      = "coco"     # keypoints are in COCO format
OUTPUT_DIR         = Path("data/court_keypoints")


def download(api_key: str) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"Accessing {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}...")
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {OUTPUT_DIR.resolve()} ...")
    dataset = version.download(EXPORT_FORMAT, location=str(OUTPUT_DIR), overwrite=True)

    print(f"\nDownload complete.")
    print(f"Location: {OUTPUT_DIR.resolve()}")

    # Quick sanity check
    image_files = list(OUTPUT_DIR.rglob("*.jpg")) + list(OUTPUT_DIR.rglob("*.png"))
    print(f"Images found: {len(image_files)}")
    if len(image_files) < 100:
        print("WARNING: fewer images than expected — check your API key and project version.")


def main():
    parser = argparse.ArgumentParser(description="Download reloc2 court keypoints dataset")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow private API key (find at app.roboflow.com → profile → API)",
    )
    args = parser.parse_args()

    # Key resolution order: argument → env var → interactive prompt
    api_key = (
        args.api_key
        or os.environ.get("ROBOFLOW_API_KEY")
        or getpass.getpass("Roboflow API key: ")
    )

    if not api_key or len(api_key) < 8:
        print("ERROR: no valid API key provided.")
        sys.exit(1)

    download(api_key)


if __name__ == "__main__":
    main()
