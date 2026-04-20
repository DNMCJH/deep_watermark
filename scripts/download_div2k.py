"""Download DIV2K training set (800 images) with resume and retry support."""

import os
import sys
import time
import hashlib
import zipfile
import urllib.request
import urllib.error

URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
ZIP_PATH = os.path.join(DEST_DIR, "DIV2K_train_HR.zip")
EXTRACT_DIR = os.path.join(DEST_DIR, "train")

EXPECTED_SIZE = 3_530_603_713
EXPECTED_MD5 = None  # set after first successful download if desired

MAX_RETRIES = 10
RETRY_DELAY_BASE = 5
BLOCK_SIZE = 1024 * 256


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url, path, expected_size=None, max_retries=MAX_RETRIES):
    """Download a file with automatic resume on failure and exponential backoff."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        existing_size = os.path.getsize(path) if os.path.exists(path) else 0

        if expected_size and existing_size >= expected_size:
            print(f"File already complete: {path}")
            return True

        try:
            req = urllib.request.Request(url)
            if existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")
                print(f"\n[Attempt {attempt}/{max_retries}] Resuming from {existing_size / (1024**2):.1f} MB")
            else:
                print(f"\n[Attempt {attempt}/{max_retries}] Starting download")

            response = urllib.request.urlopen(req, timeout=30)
            content_length = int(response.headers.get("Content-Length", 0))
            total_size = content_length + existing_size

            if expected_size:
                total_size = expected_size

            print(f"  Target: {path}")
            print(f"  Size: {total_size / (1024**2):.1f} MB total")

            mode = "ab" if existing_size > 0 else "wb"
            downloaded = existing_size

            with open(path, mode) as f:
                while True:
                    chunk = response.read(BLOCK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        bar_len = 40
                        filled = int(bar_len * pct / 100)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r  [{bar}] {downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB ({pct}%)"
                        )
                        sys.stdout.flush()

            print("\n  Download complete.")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            print(f"\n  Error: {e}")
            if attempt < max_retries:
                delay = min(RETRY_DELAY_BASE * (2 ** (attempt - 1)), 120)
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  Failed after {max_retries} attempts.")
                return False

    return False


def extract_with_resume(zip_path, extract_dir):
    """Extract images, skipping files that already exist."""
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        image_members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))]
        total = len(image_members)
        skipped = 0
        extracted = 0

        for i, member in enumerate(image_members, 1):
            filename = os.path.basename(member)
            target = os.path.join(extract_dir, filename)

            if os.path.exists(target):
                info = zf.getinfo(member)
                if os.path.getsize(target) == info.file_size:
                    skipped += 1
                    continue

            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1

            if i % 100 == 0 or i == total:
                sys.stdout.write(f"\r  Extracting: {i}/{total}")
                sys.stdout.flush()

    print(f"\n  Extracted {extracted} new, skipped {skipped} existing. Total: {total}")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    current_size = os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 0
    existing_images = []
    if os.path.isdir(EXTRACT_DIR):
        existing_images = [f for f in os.listdir(EXTRACT_DIR) if f.lower().endswith((".png", ".jpg"))]

    if len(existing_images) >= 800:
        print(f"Dataset already complete: {len(existing_images)} images in {EXTRACT_DIR}")
        return

    if current_size < EXPECTED_SIZE:
        success = download_with_resume(URL, ZIP_PATH, expected_size=EXPECTED_SIZE)
        if not success:
            print("\nDownload failed. Run this script again to resume.")
            sys.exit(1)

    print("\nVerifying zip integrity...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            bad = zf.testzip()
            if bad:
                print(f"  Corrupt file in zip: {bad}")
                print("  Deleting corrupt zip. Re-run to download again.")
                os.remove(ZIP_PATH)
                sys.exit(1)
    except zipfile.BadZipFile:
        print("  Zip file is corrupt. Deleting. Re-run to download again.")
        os.remove(ZIP_PATH)
        sys.exit(1)

    print("  Zip OK.")

    print("\nExtracting images...")
    extract_with_resume(ZIP_PATH, EXTRACT_DIR)

    count = len([f for f in os.listdir(EXTRACT_DIR) if f.lower().endswith((".png", ".jpg"))])
    print(f"\nDone. {count} images in {EXTRACT_DIR}")

    if count >= 800:
        os.remove(ZIP_PATH)
        print("Removed zip file.")
    else:
        print(f"Warning: expected 800 images but found {count}. Keeping zip for re-extraction.")


if __name__ == "__main__":
    main()
