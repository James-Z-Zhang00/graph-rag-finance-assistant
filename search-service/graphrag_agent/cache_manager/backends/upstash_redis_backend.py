import json
from typing import Any, Optional
from .base import CacheStorageBackend


class UpstashRedisCacheBackend(CacheStorageBackend):
    """Cache storage backend backed by Upstash Redis (REST API).

    Replaces the local memory+disk HybridCacheBackend when CACHE_BACKEND=upstash.
    All cache items are stored as JSON strings with an optional TTL.
    Keys are namespaced with a prefix to support multiple logical caches
    (session vs global) sharing the same Upstash Redis database.
    """

    def __init__(self, url: str, token: str, namespace: str = "cache", ttl: int = 604800):
        """
        Args:
            url:       Upstash Redis REST URL (from Upstash console).
            token:     Upstash Redis REST token (from Upstash console).
            namespace: Key prefix — use different values for session vs global cache.
            ttl:       Time-to-live in seconds (default: 7 days).
        """
        from upstash_redis import Redis
        self.redis = Redis(url=url, token=token)
        self.namespace = namespace
        self.ttl = ttl

        # Hit/miss counters (in-process only, resets on restart)
        self.hits = 0
        self.misses = 0

    def _key(self, key: str) -> str:
        """Prepend namespace to avoid collisions between session and global caches."""
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache item from Upstash Redis."""
        raw = self.redis.get(self._key(key))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any) -> None:
        """Store a cache item in Upstash Redis with TTL."""
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            print(f"[UpstashRedis] Serialization failed for key {key}: {e}")
            return
        self.redis.set(self._key(key), serialized, ex=self.ttl)

    def delete(self, key: str) -> bool:
        """Delete a cache item from Upstash Redis."""
        deleted = self.redis.delete(self._key(key))
        return deleted > 0

    def clear(self) -> None:
        """Delete all keys under this namespace using SCAN."""
        cursor = 0
        pattern = f"{self.namespace}:*"
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break

    def get_stats(self) -> dict:
        """Return hit/miss statistics for this backend instance."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0
        }
