import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from sentence_transformers import SentenceTransformer
import threading
from pathlib import Path

from graphrag_agent.config.settings import (
    MODEL_CACHE_DIR,
    CACHE_EMBEDDING_PROVIDER,
    CACHE_SENTENCE_TRANSFORMER_MODEL,
)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding vector providers."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into embedding vectors."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider that reuses the RAG embedding model."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton — avoid creating multiple instances."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Import and reuse the existing embedding model
        try:
            from graphrag_agent.models.get_models import get_embeddings_model
            self.model = get_embeddings_model()
            self._dimension = None
            self._initialized = True
        except ImportError as e:
            raise ImportError(f"Failed to import embedding model: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into vectors using the OpenAI embedding model."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Use a simple text to probe the dimension
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


class SentenceTransformerEmbedding(EmbeddingProvider):
    """SentenceTransformer-based embedding provider with model caching."""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        """Per-model singleton — avoid reloading the same model."""
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = super().__new__(cls)
                cls._instances[model_name]._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.model_name = model_name

        # Set model cache directory
        if cache_dir is None:
            cache_dir = MODEL_CACHE_DIR

        # Ensure the cache directory exists
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Load model with the specified cache directory
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_path))
        self._dimension = None
        self._initialized = True

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into normalised embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Use a simple text to probe the dimension
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


def get_cache_embedding_provider() -> EmbeddingProvider:
    """Return the configured cache embedding provider."""
    provider_type = CACHE_EMBEDDING_PROVIDER

    if provider_type == 'openai':
        return OpenAIEmbeddingProvider()
    else:
        # Use SentenceTransformer
        model_name = CACHE_SENTENCE_TRANSFORMER_MODEL
        return SentenceTransformerEmbedding(model_name=model_name, cache_dir=MODEL_CACHE_DIR)
