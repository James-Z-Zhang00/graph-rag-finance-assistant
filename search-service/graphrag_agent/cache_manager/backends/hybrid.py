from typing import Any, Optional, Set
from .base import CacheStorageBackend
from .memory import MemoryCacheBackend
from .disk import DiskCacheBackend


class HybridCacheBackend(CacheStorageBackend):
    """Hybrid cache backend — memory (L1) + configurable L2 (disk by default, or any CacheStorageBackend)."""

    def __init__(self, cache_dir: str = "./cache", memory_max_size: int = 100, disk_max_size: int = 1000,
                 l2_backend: Optional[CacheStorageBackend] = None):
        """Initialize the hybrid cache backend.

        Args:
            l2_backend: Optional L2 backend override. Defaults to DiskCacheBackend.
                        Pass an UpstashRedisCacheBackend here to use Redis as L2.
        """
        self.memory_cache = MemoryCacheBackend(max_size=memory_max_size)
        self.disk_cache = l2_backend or DiskCacheBackend(cache_dir=cache_dir, max_size=disk_max_size)
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        self.frequent_keys: Set[str] = set()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item — checks memory first, then disk."""
        value = self.memory_cache.get(key)
        if value is not None:
            self.memory_hits += 1
            return value

        value = self.disk_cache.get(key)
        if value is not None:
            self.disk_hits += 1

            # Promote high-quality items to memory cache
            is_high_quality = False
            if isinstance(value, dict) and "metadata" in value:
                metadata = value.get("metadata", {})
                is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)

            if is_high_quality:
                self.memory_cache.set(key, value)

            return value

        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store a cache item in both memory and disk."""
        is_high_quality = False
        if isinstance(value, dict) and "metadata" in value:
            metadata = value.get("metadata", {})
            is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)

        # Always write to disk
        self.disk_cache.set(key, value)

        # Always write to memory (high-quality items get priority on eviction)
        self.memory_cache.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete a cache item from both memory and disk."""
        memory_success = self.memory_cache.delete(key)
        disk_success = self.disk_cache.delete(key)
        return memory_success or disk_success

    def clear(self) -> None:
        """Clear all cached items from both memory and disk."""
        self.memory_cache.clear()
        self.disk_cache.clear()
