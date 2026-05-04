"""
GCS storage helper.

Downloads all objects under a bucket prefix to a local directory so
sec-parser can read them as if they were local files.
"""

import os
import time
from pathlib import Path

_GCS_DOWNLOAD_TIMEOUT = 300   # seconds per blob
_GCS_DOWNLOAD_RETRIES = 3


def download_prefix_to_dir(bucket_name: str, prefix: str, dest_dir: str) -> int:
    """
    Download every object under `prefix` in `bucket_name` to `dest_dir`.
    Returns the number of files downloaded.
    Preserves sub-folder structure relative to the prefix.
    Each blob is retried up to 3 times with exponential backoff on transient errors.
    """
    from google.cloud import storage

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    client = storage.Client()

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    count = 0
    for blob in blobs:
        # Skip folder placeholder objects (keys ending with /)
        if blob.name.endswith("/"):
            continue

        # Relative path inside dest_dir
        relative = blob.name[len(prefix):]
        local_path = Path(dest_dir) / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(_GCS_DOWNLOAD_RETRIES):
            try:
                blob.download_to_filename(str(local_path), timeout=_GCS_DOWNLOAD_TIMEOUT)
                break
            except Exception as exc:
                if attempt == _GCS_DOWNLOAD_RETRIES - 1:
                    raise
                wait = 5 * (2 ** attempt)  # 5s, 10s
                print(f"[gcs] download failed for {blob.name} ({exc}), retry {attempt + 1}/{_GCS_DOWNLOAD_RETRIES - 1} in {wait}s")
                time.sleep(wait)

        print(f"[gcs] downloaded gs://{bucket_name}/{blob.name} → {local_path}")
        count += 1

    return count
