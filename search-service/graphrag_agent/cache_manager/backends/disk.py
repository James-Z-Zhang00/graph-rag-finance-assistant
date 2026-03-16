import os
import time
import json
import threading
from typing import Any, Optional, List, Tuple, Dict
from collections import OrderedDict
from .base import CacheStorageBackend


class DiskCacheBackend(CacheStorageBackend):
    """Disk-based cache backend with batched writes and LRU eviction."""

    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000,
                 batch_size: int = 10, flush_interval: float = 30.0):
        """Initialize the disk cache backend."""
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # OrderedDict maintains insertion/access order for LRU
        self.metadata: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.write_queue: List[Tuple[str, Any]] = []
        self.last_flush_time = time.time()
        self._lock = threading.RLock()

        os.makedirs(cache_dir, exist_ok=True)
        self._load_index()

    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key, using 2-char subdirectories to avoid large flat dirs."""
        subdir = key[:2]
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, f"{key}.json")

    def _get_index_path(self) -> str:
        """Get the path to the index file."""
        return os.path.join(self.cache_dir, "index.json")

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key in sorted(data.keys(), key=lambda k: data[k].get('last_accessed', 0)):
                        self.metadata[key] = data[key]
            except Exception as e:
                print(f"Failed to load cache index: {e}")
                self.metadata = OrderedDict()

        self._sync_index_with_filesystem()

    def _sync_index_with_filesystem(self) -> None:
        """Sync the index with actual files on disk."""
        existing_files = set()

        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if os.path.isdir(item_path) and len(item) == 2:
                for filename in os.listdir(item_path):
                    if filename.endswith(".json"):
                        key = filename[:-5]  # strip .json
                        existing_files.add(key)

        # Remove index entries with no corresponding file
        keys_to_remove = [key for key in self.metadata if key not in existing_files]
        for key in keys_to_remove:
            del self.metadata[key]

        # Add files that exist on disk but are missing from the index
        for key in existing_files:
            if key not in self.metadata:
                file_path = self._get_cache_path(key)
                try:
                    stat = os.stat(file_path)
                    self.metadata[key] = {
                        "created_at": stat.st_ctime,
                        "last_accessed": stat.st_atime,
                        "access_count": 0,
                        "file_size": stat.st_size
                    }
                except OSError:
                    continue

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self._get_index_path(), 'w', encoding='utf-8') as f:
                json.dump(dict(self.metadata), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache index: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item from disk."""
        with self._lock:
            cache_path = self._get_cache_path(key)

            if key in self.metadata and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        value = json.load(f)

                    self.metadata[key]["last_accessed"] = time.time()
                    self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                    self.metadata.move_to_end(key)
                    self._schedule_index_save()

                    return value
                except Exception as e:
                    print(f"Failed to read cache file ({key}): {e}")
                    if key in self.metadata:
                        del self.metadata[key]

            return None

    def set(self, key: str, value: Any) -> None:
        """Write a cache item, evicting items if at capacity."""
        with self._lock:
            if len(self.metadata) >= self.max_size and key not in self.metadata:
                self._evict_items()

            current_time = time.time()
            if key in self.metadata:
                self.metadata[key].update({
                    "last_accessed": current_time,
                    "access_count": self.metadata[key].get("access_count", 0)
                })
                self.metadata.move_to_end(key)
            else:
                self.metadata[key] = {
                    "created_at": current_time,
                    "last_accessed": current_time,
                    "access_count": 0
                }

            self.write_queue.append((key, value))

            if (len(self.write_queue) >= self.batch_size or
                    (time.time() - self.last_flush_time) > self.flush_interval):
                self._flush_write_queue()

    def _flush_write_queue(self) -> None:
        """Flush all pending writes to disk."""
        if not self.write_queue:
            return

        successful_writes = []
        failed_writes = []

        for key, value in self.write_queue:
            try:
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2, default=str)

                if key in self.metadata:
                    self.metadata[key]["file_size"] = os.path.getsize(cache_path)

                successful_writes.append(key)
            except Exception as e:
                print(f"Failed to write cache file ({key}): {e}")
                failed_writes.append((key, value))

        # Keep only failed writes in the queue for retry
        self.write_queue = failed_writes
        self.last_flush_time = time.time()

        if successful_writes:
            self._save_index()

    def _schedule_index_save(self) -> None:
        """Save the index at most once per minute."""
        current_time = time.time()
        if current_time - self.last_flush_time > 60:
            self._save_index()
            self.last_flush_time = current_time

    def delete(self, key: str) -> bool:
        """Delete a cache item from disk and index."""
        with self._lock:
            if key not in self.metadata:
                return False

            del self.metadata[key]

            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"Failed to delete cache file ({key}): {e}")
                    return False

            self.write_queue = [(k, v) for k, v in self.write_queue if k != key]
            self._save_index()
            return True

    def clear(self) -> None:
        """Clear all cached items from disk and memory."""
        with self._lock:
            self.write_queue.clear()
            self.metadata.clear()

            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith(".json"):
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception as e:
                            print(f"Failed to delete cache file: {e}")

            self._save_index()

    def flush(self) -> None:
        """Force flush all pending writes to disk."""
        with self._lock:
            self._flush_write_queue()

    def _evict_items(self, num_to_evict: int = None) -> None:
        """Evict cache items using a composite score of frequency, recency, and size."""
        if not self.metadata:
            return

        if num_to_evict is None:
            num_to_evict = max(1, len(self.metadata) // 10)  # evict 10%

        current_time = time.time()
        scores = {}

        for key, meta in self.metadata.items():
            age = current_time - meta.get("created_at", current_time)
            access_count = meta.get("access_count", 0)
            last_accessed = meta.get("last_accessed", meta.get("created_at", current_time))
            recency = current_time - last_accessed
            file_size = meta.get("file_size", 1000)  # default 1KB

            frequency_score = access_count / max(age / 3600, 1)  # accesses per hour
            recency_score = 1 / max(recency / 3600, 1)
            size_penalty = file_size / 1024  # penalise large files

            scores[key] = frequency_score + recency_score - size_penalty * 0.1

        # Evict lowest-scored items
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[:num_to_evict]
        for key in keys_to_evict:
            self.delete(key)
