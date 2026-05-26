from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


import os

from graphrag_agent.config.settings import (
    TIKTOKEN_CACHE_DIR,
    OPENAI_EMBEDDING_CONFIG,
    OPENAI_LLM_CONFIG,
    EXTRACTION_LLM_CONFIG,
    DEDUP_LLM_CONFIG,
    COMMUNITY_LLM_CONFIG,
    OPENAI_BASE_URL,
)


# Set tiktoken cache directory to avoid downloading it on every run
def setup_cache():
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


setup_cache()


def _make_oidc_http_client():
    """
    Return an httpx.Client that fetches a fresh Google OIDC token for every
    request when running on Cloud Run. Returns None locally (no auth needed).

    Using a per-request token avoids the 1-hour expiry that bites long builds
    when the token is fetched once at model creation and reused for all LLM calls.
    """
    if not os.getenv("K_SERVICE"):
        return None
    try:
        import httpx
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        audience = OPENAI_BASE_URL.rstrip("/")
        if audience.endswith("/v1"):
            audience = audience[:-3]

        class _OIDCAuth(httpx.Auth):
            def auth_flow(self, request):
                token = id_token.fetch_id_token(Request(), audience)
                request.headers["Authorization"] = f"Bearer {token}"
                yield request

        return httpx.Client(auth=_OIDCAuth())
    except Exception:
        return None


def get_embeddings_model():
    config = {k: v for k, v in OPENAI_EMBEDDING_CONFIG.items() if v}
    http_client = _make_oidc_http_client()
    if http_client:
        config["http_client"] = http_client
    return OpenAIEmbeddings(**config)


def _make_llm(config_dict: dict) -> ChatOpenAI:
    config = {k: v for k, v in config_dict.items() if v is not None and v != ""}
    http_client = _make_oidc_http_client()
    if http_client:
        config["http_client"] = http_client
    return ChatOpenAI(**config)


def get_llm_model():
    return _make_llm(OPENAI_LLM_CONFIG)


def get_extraction_llm():
    """LLM for step 4: entity/relationship extraction from text chunks."""
    return _make_llm(EXTRACTION_LLM_CONFIG)


def get_dedup_llm():
    """LLM for step 5: entity deduplication, merging, and alignment."""
    return _make_llm(DEDUP_LLM_CONFIG)


def get_community_llm():
    """LLM for step 6: community summary generation."""
    return _make_llm(COMMUNITY_LLM_CONFIG)


def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    manager = AsyncCallbackManager(handlers=[callback_handler])

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config.update({"streaming": True, "callbacks": manager})
    http_client = _make_oidc_http_client()
    if http_client:
        config["http_client"] = http_client
    return ChatOpenAI(**config)

def count_tokens(text):
    """Simple general-purpose token counter."""
    if not text:
        return 0

    model_name = (OPENAI_LLM_CONFIG.get("model") or "").lower()

    # DeepSeek models — use transformers tokenizer
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
            return len(tokenizer.encode(text))
        except:
            pass

    # GPT models — use tiktoken
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            pass

    # Fallback: rough character-based estimate
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4

if __name__ == '__main__':
    # Test LLM
    llm = get_llm_model()
    print(llm.invoke("Hello"))

    # Streaming test is currently broken due to LangChain version issues
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("Hello"))

    # Test embeddings
    test_text = "Hello, this is a test."
    embeddings = get_embeddings_model()
    print(embeddings.embed_query(test_text))

    # Test token counting
    test_text = "Hello world"
    tokens = count_tokens(test_text)
    print(f"Token count: '{test_text}' = {tokens} tokens")
