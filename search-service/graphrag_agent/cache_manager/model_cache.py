"""
Model cache management — preloads and manages embedding model caches.
"""

from typing import List, Optional

from graphrag_agent.config.settings import (
    MODEL_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODELS,
    CACHE_EMBEDDING_PROVIDER,
    CACHE_SENTENCE_TRANSFORMER_MODEL,
)


def ensure_model_cache_dir() -> str:
    """Ensure the model cache directory exists and return its path."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODEL_CACHE_DIR)


def preload_sentence_transformer_models(models: Optional[List[str]] = None) -> None:
    """Preload SentenceTransformer models into the cache directory."""
    try:
        from sentence_transformers import SentenceTransformer

        if models is None:
            models = list(SENTENCE_TRANSFORMER_MODELS)

        if not models:
            return

        cache_dir = ensure_model_cache_dir()

        for model_name in models:
            try:
                _ = SentenceTransformer(model_name, cache_folder=cache_dir)
                print(f"Model {model_name} loaded successfully")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")

    except ImportError as e:
        print(f"sentence_transformers not installed, skipping preload: {e}")


def preload_cache_embedding_model() -> None:
    """Preload the embedding model used by the cache."""
    provider_type = CACHE_EMBEDDING_PROVIDER

    if provider_type == 'openai':
        # OpenAI models do not need local preloading
        return

    model_name = CACHE_SENTENCE_TRANSFORMER_MODEL
    preload_sentence_transformer_models([model_name])


def initialize_model_cache() -> None:
    """Initialize the model cache by preloading configured models."""
    ensure_model_cache_dir()
    preload_cache_embedding_model()


if __name__ == "__main__":
    # Run this script directly to preload models
    initialize_model_cache()
