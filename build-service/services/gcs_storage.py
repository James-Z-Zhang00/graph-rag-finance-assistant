"""
GCS storage helper.

Downloads all objects under a bucket prefix to a local directory so
sec-parser can read them as if they were local files.
"""

import os
from pathlib import Path


def download_prefix_to_dir(bucket_name: str, prefix: str, dest_dir: str) -> int:
    """
    Download every object under `prefix` in `bucket_name` to `dest_dir`.
    Returns the number of files downloaded.
    Preserves sub-folder structure relative to the prefix.
    """
    from google.cloud import storage

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

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

        blob.download_to_filename(str(local_path))
        print(f"[gcs] downloaded gs://{bucket_name}/{blob.name} → {local_path}")
        count += 1

    return count
