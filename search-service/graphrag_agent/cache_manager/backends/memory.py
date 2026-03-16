import time
from typing import Any, Optional
from .base import CacheStorageBackend


class MemoryCacheBackend(CacheStorageBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: Maximum number of items to hold in memory.
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}  # used for LRU eviction

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item and update its access time."""
        value = self.cache.get(key)
        if value is not None:
            self.access_times[key] = time.time()
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a cache item, evicting the LRU item if at capacity."""
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        self.cache[key] = value
        self.access_times[key] = time.time()

    def delete(self, key: str) -> bool:
        """Delete a cache item."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self.access_times:
            return
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(oldest_key)

    def cleanup_unused(self) -> None:
        """Remove stale keys from access_times that no longer exist in cache."""
        unused_keys = [k for k in self.access_times if k not in self.cache]
        for key in unused_keys:
            del self.access_times[key]
