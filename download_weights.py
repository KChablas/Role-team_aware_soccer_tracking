"""
Download model weights from GCS bucket.

Edit the constants below to match your bucket, then run:
    python download_weights.py
"""

import subprocess
import sys
from pathlib import Path

# ── Edit these ────────────────────────────────────────────────────────────────
GCS_BUCKET = "gs://soccer-tracker-weights"
WEIGHTS = [
    "best_v11_fromgcp.pt",
    "prtreid-soccernet-baseline.pth.tar",
]
LOCAL_DIR = "weights"
# ─────────────────────────────────────────────────────────────────────────────


def check_gsutil():
    result = subprocess.run(["gsutil", "version"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: gsutil not found.")
        print("Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)


def download_weight(filename, local_dir):
    local_path = Path(local_dir) / filename
    gcs_path = f"{GCS_BUCKET}/{filename}"

    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}: already exists ({size_mb:.1f} MB), skipping.")
        return

    print(f"  {filename}: downloading from {gcs_path}...")
    result = subprocess.run(["gsutil", "cp", gcs_path, str(local_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: download failed for {filename}")
        print(f"  {result.stderr.strip()}")
        print(f"  Check that GCS_BUCKET='{GCS_BUCKET}' is correct and you have read access.")
        sys.exit(1)

    if not local_path.exists():
        print(f"  ERROR: {filename} not found after download.")
        sys.exit(1)

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  {filename}: done ({size_mb:.1f} MB)")


def main():
    check_gsutil()
    Path(LOCAL_DIR).mkdir(exist_ok=True)

    for filename in WEIGHTS:
        download_weight(filename, LOCAL_DIR)

    print(f"\nAll weights downloaded to {LOCAL_DIR}/")


if __name__ == "__main__":
    main()
