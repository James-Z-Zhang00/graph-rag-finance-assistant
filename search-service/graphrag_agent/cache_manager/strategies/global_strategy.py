import hashlib
from .base import CacheKeyStrategy


class GlobalCacheKeyStrategy(CacheKeyStrategy):
    """Global cache key strategy — ignores session ID and all context parameters."""

    def generate_key(self, query: str, **kwargs) -> str:
        """
        Generate a key using only the query content, ignoring thread ID and other context.

        Args:
            query: The query string.
            **kwargs: Additional parameters (ignored).

        Returns:
            str: The generated cache key.
        """
        # Strip any prefix (e.g. "generate:")
        if ":" in query:
            parts = query.split(":", 1)
            if len(parts) > 1:
                query = parts[1]

        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()
