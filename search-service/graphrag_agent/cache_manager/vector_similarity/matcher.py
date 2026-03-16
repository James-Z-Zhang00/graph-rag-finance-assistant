import faiss
import pickle
import os
import threading
from typing import List, Tuple, Dict, Any
from .embeddings import EmbeddingProvider, get_cache_embedding_provider

from graphrag_agent.config.settings import similarity_threshold as st


class VectorSimilarityMatcher:
    """Vector similarity matcher for semantic cache lookups using FAISS."""

    def __init__(self,
                 embedding_provider: EmbeddingProvider = None,
                 similarity_threshold: float = st,
                 max_vectors: int = 10000,
                 index_file: str = None):
        """
        Args:
            embedding_provider: Embedding provider; auto-selected from config if None.
            similarity_threshold: Minimum cosine similarity to count as a match.
            max_vectors: Maximum number of vectors to hold in the index.
            index_file: Path prefix for persisting the index to disk.
        """
        self.embedding_provider = embedding_provider or get_cache_embedding_provider()
        self.similarity_threshold = similarity_threshold
        self.max_vectors = max_vectors
        self.index_file = index_file

        # Initialise FAISS inner-product (cosine) index
        self.dimension = self.embedding_provider.get_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)

        # Mappings between cache keys and FAISS slot indices
        self.key_to_index = {}
        self.index_to_key = {}
        self.key_to_context = {}
        self.key_to_query = {}  # stores the original query string per key

        self._lock = threading.RLock()
        self._next_index = 0

        # Load persisted index if available
        if self.index_file and os.path.exists(f"{self.index_file}.pkl"):
            self._load_index()

    def add_vector(self, cache_key: str, query: str, context_info: Dict[str, Any] = None):
        """Add a query embedding to the index under cache_key."""
        with self._lock:
            # Replace existing entry if present
            if cache_key in self.key_to_index:
                self.remove_vector(cache_key)

            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            faiss_index = self._next_index
            self.index.add(embedding)

            self.key_to_index[cache_key] = faiss_index
            self.index_to_key[faiss_index] = cache_key
            self.key_to_context[cache_key] = context_info or {}
            self.key_to_query[cache_key] = query

            self._next_index += 1

            if self._next_index > self.max_vectors:
                self._cleanup_old_vectors()

    def find_similar(self, query: str, context_info: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k cache keys whose stored queries are most similar to query."""
        with self._lock:
            if self.index.ntotal == 0:
                return []

            query_embedding = self.embedding_provider.encode(query)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.index_to_key):
                    continue

                if idx in self.index_to_key:
                    cache_key = self.index_to_key[idx]

                    if self._context_matches(context_info, self.key_to_context.get(cache_key, {})):
                        if score >= self.similarity_threshold:
                            results.append((cache_key, float(score)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def remove_vector(self, cache_key: str):
        """Remove a cache key's vector from the index."""
        with self._lock:
            if cache_key not in self.key_to_index:
                return

            faiss_index = self.key_to_index[cache_key]

            del self.key_to_index[cache_key]
            if faiss_index in self.index_to_key:
                del self.index_to_key[faiss_index]
            if cache_key in self.key_to_context:
                del self.key_to_context[cache_key]
            if cache_key in self.key_to_query:
                del self.key_to_query[cache_key]

    def clear(self):
        """Clear all vectors from the index."""
        with self._lock:
            self.index.reset()
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0

    def _context_matches(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> bool:
        """Return True if both contexts belong to the same thread."""
        if not context1 and not context2:
            return True

        if not context1 or not context2:
            return False

        thread_id1 = context1.get('thread_id', 'default')
        thread_id2 = context2.get('thread_id', 'default')

        return thread_id1 == thread_id2

    def _cleanup_old_vectors(self):
        """Evict old vectors to stay within max_vectors capacity."""
        # Rebuild the index keeping only the most recent vectors
        pass

    def save_index(self, file_path: str = None):
        """Persist the FAISS index and key mappings to disk."""
        if file_path is None:
            file_path = self.index_file

        if file_path is None:
            return

        with self._lock:
            try:
                data = {
                    'key_to_index': self.key_to_index,
                    'index_to_key': self.index_to_key,
                    'key_to_context': self.key_to_context,
                    'key_to_query': self.key_to_query,
                    'next_index': self._next_index
                }

                # Save FAISS index
                if self.index.ntotal > 0:
                    faiss.write_index(self.index, f"{file_path}.faiss")

                # Save key mappings
                with open(f"{file_path}.pkl", 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Failed to save vector index: {e}")

    def _load_index(self):
        """Load the FAISS index and key mappings from disk."""
        try:
            with open(f"{self.index_file}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.key_to_index = data.get('key_to_index', {})
                self.index_to_key = data.get('index_to_key', {})
                self.key_to_context = data.get('key_to_context', {})
                self.key_to_query = data.get('key_to_query', {})
                self._next_index = data.get('next_index', 0)

            faiss_file = f"{self.index_file}.faiss"
            if os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
            else:
                # FAISS file missing — rebuild from stored queries
                self._rebuild_index()

        except Exception as e:
            print(f"Failed to load vector index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0

    def _rebuild_index(self):
        """Rebuild the FAISS index from stored query strings."""
        if not self.key_to_query:
            return

        self.index = faiss.IndexFlatIP(self.dimension)

        for cache_key, query in self.key_to_query.items():
            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            self.index.add(embedding)