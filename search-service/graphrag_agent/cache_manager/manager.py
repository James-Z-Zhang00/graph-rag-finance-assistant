import time
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from .strategies import CacheKeyStrategy, SimpleCacheKeyStrategy, ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy
from .backends import CacheStorageBackend, MemoryCacheBackend, HybridCacheBackend, ThreadSafeCacheBackend
from .models import CacheItem
from .vector_similarity import VectorSimilarityMatcher, get_cache_embedding_provider

from graphrag_agent.config.settings import CACHE_SETTINGS


class CacheManager:
    """Unified cache manager with vector similarity matching."""

    def __init__(self,
                 key_strategy: CacheKeyStrategy = None,
                 storage_backend: CacheStorageBackend = None,
                 cache_dir: Optional[str] = None,
                 memory_only: Optional[bool] = None,
                 max_memory_size: Optional[int] = None,
                 max_disk_size: Optional[int] = None,
                 thread_safe: Optional[bool] = None,
                 enable_vector_similarity: Optional[bool] = None,
                 similarity_threshold: Optional[float] = None,
                 max_vectors: Optional[int] = None):
        """
        Args:
            key_strategy: Cache key generation strategy.
            storage_backend: Storage backend instance.
            cache_dir: Directory for disk cache.
            memory_only: Use in-memory storage only.
            max_memory_size: Maximum number of items in memory cache.
            max_disk_size: Maximum number of items in disk cache.
            thread_safe: Wrap the backend with a thread-safe decorator.
            enable_vector_similarity: Enable semantic (vector) cache lookup.
            similarity_threshold: Minimum cosine similarity for a vector hit.
            max_vectors: Maximum vectors held in the FAISS index.
        """
        # Set cache key strategy
        self.key_strategy = key_strategy or SimpleCacheKeyStrategy()

        # Load defaults from unified config
        cache_config = CACHE_SETTINGS
        cache_dir = cache_dir or str(cache_config["dir"])
        memory_only = cache_config["memory_only"] if memory_only is None else memory_only
        max_memory_size = cache_config["max_memory_size"] if max_memory_size is None else max_memory_size
        max_disk_size = cache_config["max_disk_size"] if max_disk_size is None else max_disk_size
        thread_safe = cache_config["thread_safe"] if thread_safe is None else thread_safe
        enable_vector_similarity = (
            cache_config["enable_vector_similarity"]
            if enable_vector_similarity is None
            else enable_vector_similarity
        )
        similarity_threshold = (
            cache_config["similarity_threshold"]
            if similarity_threshold is None
            else similarity_threshold
        )
        max_vectors = cache_config["max_vectors"] if max_vectors is None else max_vectors
        
        # Build storage backend
        backend = self._create_storage_backend(
            storage_backend, memory_only, cache_dir,
            max_memory_size, max_disk_size
        )

        # Wrap with thread-safety decorator if requested
        self.storage = ThreadSafeCacheBackend(backend) if thread_safe else backend

        # Vector similarity matcher
        self.enable_vector_similarity = enable_vector_similarity
        if enable_vector_similarity:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            vector_index_file = f"{cache_dir}/vector_index" if not memory_only else None

            embedding_provider = get_cache_embedding_provider()

            self.vector_matcher = VectorSimilarityMatcher(
                embedding_provider=embedding_provider,
                similarity_threshold=similarity_threshold,
                max_vectors=max_vectors,
                index_file=vector_index_file
            )
        else:
            self.vector_matcher = None

        # Performance metrics
        self.performance_metrics = {
            'exact_hits': 0,
            'vector_hits': 0,
            'misses': 0,
            'total_queries': 0
        }
    
    def _create_storage_backend(self, storage_backend, memory_only, cache_dir,
                               max_memory_size, max_disk_size) -> CacheStorageBackend:
        """Create and return the appropriate storage backend."""
        if storage_backend:
            return storage_backend
        elif memory_only:
            return MemoryCacheBackend(max_size=max_memory_size)
        else:
            return HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=max_memory_size,
                disk_max_size=max_disk_size
            )
    
    def _get_consistent_key(self, query: str, **kwargs) -> str:
        """Generate a deterministic cache key for the given query."""
        return self.key_strategy.generate_key(query, **kwargs)

    def _extract_context_info(self, **kwargs) -> Dict[str, Any]:
        """Extract context fields used for vector similarity matching."""
        return {
            'thread_id': kwargs.get('thread_id', 'default'),
            'keywords': kwargs.get('keywords', []),
            'low_level_keywords': kwargs.get('low_level_keywords', []),
            'high_level_keywords': kwargs.get('high_level_keywords', [])
        }
    
    def get(self, query: str, skip_validation: bool = False, **kwargs) -> Optional[Any]:
        """Retrieve cached content — exact match first, then vector similarity fallback."""
        start_time = time.time()
        self.performance_metrics['total_queries'] += 1

        key = self._get_consistent_key(query, **kwargs)

        # Exact match
        cached_data = self.storage.get(key)
        if cached_data is not None:
            self.performance_metrics['exact_hits'] += 1
            cache_item = CacheItem.from_any(cached_data)
            cache_item.update_access_stats()

            content = cache_item.get_content()
            self.performance_metrics["get_time"] = time.time() - start_time
            return content

        # Vector similarity fallback
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=3)

            for similar_key, similarity_score in similar_keys:
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    self.performance_metrics['vector_hits'] += 1
                    cache_item = CacheItem.from_any(cached_data)
                    cache_item.update_access_stats()

                    cache_item.metadata['similarity_score'] = similarity_score
                    cache_item.metadata['original_query'] = query
                    cache_item.metadata['matched_via_vector'] = True

                    content = cache_item.get_content()
                    self.performance_metrics["get_time"] = time.time() - start_time
                    return content

        # Cache miss
        self.performance_metrics['misses'] += 1
        self.performance_metrics["get_time"] = time.time() - start_time
        return None
    
    def get_fast(self, query: str, **kwargs) -> Optional[Any]:
        """Return cached content only if it is marked high-quality."""
        start_time = time.time()

        key = self._get_consistent_key(query, **kwargs)

        cached_data = self.storage.get(key)
        if cached_data is not None:
            cache_item = CacheItem.from_any(cached_data)

            if cache_item.is_high_quality():
                cache_item.update_access_stats()
                self._update_strategy_history(query, **kwargs)

                content = cache_item.get_content()
                self.performance_metrics["fast_get_time"] = time.time() - start_time
                return content

        # Try vector similarity for a high-quality hit
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=1)

            for similar_key, similarity_score in similar_keys:
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    cache_item = CacheItem.from_any(cached_data)

                    if cache_item.is_high_quality():
                        cache_item.update_access_stats()
                        cache_item.metadata['similarity_score'] = similarity_score
                        cache_item.metadata['matched_via_vector'] = True

                        content = cache_item.get_content()
                        self.performance_metrics["fast_get_time"] = time.time() - start_time
                        return content

        self.performance_metrics["fast_get_time"] = time.time() - start_time
        return None
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """Store a result in the cache and add it to the vector index."""
        start_time = time.time()

        self._update_strategy_history(query, **kwargs)

        key = self._get_consistent_key(query, **kwargs)
        cache_item = self._wrap_cache_item(result)
        self.storage.set(key, cache_item.to_dict())

        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            self.vector_matcher.add_vector(key, query, context_info)

        self.performance_metrics["set_time"] = time.time() - start_time

    def _update_strategy_history(self, query: str, **kwargs):
        """Update the key strategy's conversation history."""
        if isinstance(self.key_strategy, (ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy)):
            thread_id = kwargs.get("thread_id", "default")
            self.key_strategy.update_history(query, thread_id)
    
    def _wrap_cache_item(self, result: Any) -> CacheItem:
        """Wrap a raw result in a CacheItem."""
        if isinstance(result, dict) and "content" in result and "metadata" in result:
            return CacheItem.from_dict(result)
        else:
            return CacheItem(result)
    
    def mark_quality(self, query: str, is_positive: bool, **kwargs) -> bool:
        """Apply a positive or negative quality signal to a cached item."""
        start_time = time.time()

        key = self._get_consistent_key(query, **kwargs)

        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["mark_time"] = time.time() - start_time
            return False

        cache_item = CacheItem.from_any(cached_data)
        cache_item.mark_quality(is_positive)

        item_dict = cache_item.to_dict()
        if is_positive and cache_item.is_high_quality():
            item_dict["metadata"]["fast_path_eligible"] = True

        self.storage.set(key, item_dict)

        self.performance_metrics["mark_time"] = time.time() - start_time
        return True

    def delete(self, query: str, **kwargs) -> bool:
        """Delete a cached item and its vector index entry."""
        key = self._get_consistent_key(query, **kwargs)

        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.remove_vector(key)

        return self.storage.delete(key)

    def clear(self) -> None:
        """Clear all cached items and the vector index."""
        self.storage.clear()
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.clear()
    
    def flush(self) -> None:
        """Force all pending writes to disk and persist the vector index."""
        if hasattr(self.storage, 'backend') and hasattr(self.storage.backend, 'flush'):
            self.storage.backend.flush()
        elif hasattr(self.storage, 'flush'):
            self.storage.flush()

        # For hybrid backends, also flush the inner disk cache
        if hasattr(self.storage, 'backend'):
            backend = self.storage.backend
            if hasattr(backend, 'disk_cache') and hasattr(backend.disk_cache, 'flush'):
                backend.disk_cache.flush()
        elif hasattr(self.storage, 'disk_cache') and hasattr(self.storage.disk_cache, 'flush'):
            self.storage.disk_cache.flush()

        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of performance metrics."""
        metrics = self.performance_metrics.copy()
        if metrics['total_queries'] > 0:
            metrics['exact_hit_rate'] = metrics['exact_hits'] / metrics['total_queries']
            metrics['vector_hit_rate'] = metrics['vector_hits'] / metrics['total_queries']
            metrics['total_hit_rate'] = (metrics['exact_hits'] + metrics['vector_hits']) / metrics['total_queries']
            metrics['miss_rate'] = metrics['misses'] / metrics['total_queries']
        return metrics
    
    def validate_answer(self, query: str, answer: str, validator: Callable[[str, str], bool] = None, **kwargs) -> bool:
        """Validate whether an answer is acceptable for a given query."""
        key = self.key_strategy.generate_key(query, **kwargs)

        cached_data = self.storage.get(key)
        if cached_data is None:
            if validator:
                return validator(query, answer)
            return self._default_validation(query, answer)

        cache_item = CacheItem.from_any(cached_data)

        if cache_item.metadata.get("user_verified", False):
            return True

        quality_score = cache_item.metadata.get("quality_score", 0)
        if quality_score < 0:
            return False

        if validator:
            return validator(query, answer)

        return self._default_validation(query, answer)

    def _default_validation(self, query: str, answer: str) -> bool:
        """Basic answer validation: length check and query-word overlap."""
        if len(answer.strip()) < 10:
            return False

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        common_words = query_words.intersection(answer_words)
        if len(common_words) == 0 and len(query_words) > 2:
            return False

        return True

    def save_vector_index(self):
        """Persist the vector index to disk."""
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()
