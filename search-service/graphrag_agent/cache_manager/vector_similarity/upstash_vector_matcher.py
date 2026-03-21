import threading
from typing import List, Tuple, Dict, Any, Optional
from .embeddings import EmbeddingProvider, get_cache_embedding_provider

from graphrag_agent.config.settings import similarity_threshold as default_threshold


class UpstashVectorMatcher:
    """Vector similarity matcher backed by Upstash Vector (REST API).

    Replaces the local FAISS-based VectorSimilarityMatcher when CACHE_BACKEND=upstash.
    Implements the same interface so CacheManager works without modification:
        add_vector(), find_similar(), remove_vector(), clear(), save_index()

    Vectors are stored in Upstash Vector with metadata (query text, thread_id)
    for context-aware filtering. Upstash persists automatically — save_index() is a no-op.
    """

    def __init__(self,
                 url: str,
                 token: str,
                 namespace: str = "cache",
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 similarity_threshold: float = default_threshold,
                 max_vectors: int = 10000):
        """
        Args:
            url:                  Upstash Vector REST URL (from Upstash console).
            token:                Upstash Vector REST token (from Upstash console).
            namespace:            Prefix for vector IDs — separates session vs global vectors.
            embedding_provider:   Embedding model; auto-selected from config if None.
            similarity_threshold: Minimum cosine similarity score to count as a match.
            max_vectors:          Not enforced server-side; kept for interface compatibility.
        """
        from upstash_vector import Index
        self.index = Index(url=url, token=token)
        self.namespace = namespace
        self.embedding_provider = embedding_provider or get_cache_embedding_provider()
        self.similarity_threshold = similarity_threshold
        self.max_vectors = max_vectors
        self._lock = threading.RLock()

    def _vector_id(self, cache_key: str) -> str:
        """Namespace the cache key to avoid collisions between session and global indexes."""
        return f"{self.namespace}:{cache_key}"

    def add_vector(self, cache_key: str, query: str, context_info: Dict[str, Any] = None) -> None:
        """Encode query and upsert its vector into Upstash Vector."""
        with self._lock:
            embedding = self.embedding_provider.encode(query)
            if hasattr(embedding, 'tolist'):
                vector = embedding.tolist()
            else:
                vector = list(embedding)

            metadata = {
                "query": query,
                "thread_id": (context_info or {}).get("thread_id", "default"),
                "cache_key": cache_key
            }

            try:
                self.index.upsert(vectors=[(self._vector_id(cache_key), vector, metadata)])
            except Exception as e:
                print(f"[UpstashVector] Failed to upsert vector for key {cache_key}: {e}")

    def find_similar(self, query: str, context_info: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Query Upstash Vector for the most semantically similar cached queries.

        Returns a list of (cache_key, similarity_score) tuples filtered by
        similarity_threshold and thread_id context matching.
        """
        with self._lock:
            embedding = self.embedding_provider.encode(query)
            if hasattr(embedding, 'tolist'):
                vector = embedding.tolist()
            else:
                vector = list(embedding)

            query_thread_id = (context_info or {}).get("thread_id", "default")

            try:
                results = self.index.query(
                    vector=vector,
                    top_k=min(top_k * 2, 20),  # fetch extra to allow for thread filtering
                    include_metadata=True
                )
            except Exception as e:
                print(f"[UpstashVector] Query failed: {e}")
                return []

            matches = []
            for result in results:
                score = result.score
                if score < self.similarity_threshold:
                    continue

                metadata = result.metadata or {}
                result_thread_id = metadata.get("thread_id", "default")

                # Only return results from the same conversation thread
                if result_thread_id != query_thread_id:
                    continue

                # Extract the original cache_key from metadata
                cache_key = metadata.get("cache_key", "")
                if not cache_key:
                    # Fallback: strip namespace prefix from vector ID
                    vid = result.id or ""
                    prefix = f"{self.namespace}:"
                    cache_key = vid[len(prefix):] if vid.startswith(prefix) else vid

                matches.append((cache_key, float(score)))

            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:top_k]

    def remove_vector(self, cache_key: str) -> None:
        """Delete a vector from Upstash Vector by cache key."""
        with self._lock:
            try:
                self.index.delete(ids=[self._vector_id(cache_key)])
            except Exception as e:
                print(f"[UpstashVector] Failed to delete vector for key {cache_key}: {e}")

    def clear(self) -> None:
        """Reset the Upstash Vector index — removes all vectors."""
        with self._lock:
            try:
                self.index.reset()
            except Exception as e:
                print(f"[UpstashVector] Failed to reset index: {e}")

    def save_index(self) -> None:
        """No-op — Upstash Vector persists automatically."""
        pass
