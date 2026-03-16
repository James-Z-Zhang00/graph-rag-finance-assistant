from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheStorageBackend(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store a cache item."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache item."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items."""
        pass
