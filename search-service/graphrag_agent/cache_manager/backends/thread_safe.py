import threading
from typing import Any, Optional
from .base import CacheStorageBackend


class ThreadSafeCacheBackend(CacheStorageBackend):
    """Thread-safe decorator for any cache storage backend."""

    def __init__(self, backend: CacheStorageBackend):
        """
        Args:
            backend: The cache backend to wrap with thread safety.
        """
        self.backend = backend
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item."""
        with self.lock:
            return self.backend.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a cache item."""
        with self.lock:
            self.backend.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete a cache item."""
        with self.lock:
            return self.backend.delete(key)

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.backend.clear()
