"""Download DIV2K training set (800 images)."""

import os
import zipfile
import urllib.request
import sys

URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
ZIP_PATH = os.path.join(DEST_DIR, "DIV2K_train_HR.zip")
EXTRACT_DIR = os.path.join(DEST_DIR, "train")


def download_with_resume(url, path):
    """Download with resume support for interrupted transfers."""
    existing_size = os.path.getsize(path) if os.path.exists(path) else 0

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        print(f"Resuming from {existing_size / (1024**2):.1f} MB")

    response = urllib.request.urlopen(req)
    total_size = int(response.headers.get("Content-Length", 0)) + existing_size
    print(f"Downloading {url}")
    print(f"Saving to {path} ({total_size / (1024**2):.1f} MB total)")

    mode = "ab" if existing_size > 0 else "wb"
    downloaded = existing_size
    block_size = 1024 * 256

    with open(path, mode) as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                sys.stdout.write(f"\r  {downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB ({pct}%)")
                sys.stdout.flush()

    print("\nDownload complete.")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    expected_size = 3530603713
    current_size = os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 0

    if current_size < expected_size:
        download_with_resume(URL, ZIP_PATH)
    else:
        print(f"ZIP already complete: {ZIP_PATH}")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # Extract images directly into dataset/train/
        for member in zf.namelist():
            if member.endswith((".png", ".jpg", ".jpeg")):
                filename = os.path.basename(member)
                target = os.path.join(EXTRACT_DIR, filename)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    count = len([f for f in os.listdir(EXTRACT_DIR) if f.endswith((".png", ".jpg"))])
    print(f"Done. {count} images in {EXTRACT_DIR}")

    # Clean up zip
    os.remove(ZIP_PATH)
    print("Removed zip file.")


if __name__ == "__main__":
    main()
