from abc import ABC, abstractmethod


class CacheKeyStrategy(ABC):
    """Abstract base class for cache key generation strategies."""

    @abstractmethod
    def generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key."""
        pass
