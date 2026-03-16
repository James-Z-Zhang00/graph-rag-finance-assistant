import hashlib
from .base import CacheKeyStrategy


class SimpleCacheKeyStrategy(CacheKeyStrategy):
    """Simple MD5 hash cache key strategy."""

    def generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key using the MD5 hash of the query string."""
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()
