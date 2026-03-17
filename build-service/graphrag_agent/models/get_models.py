from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


import os

from graphrag_agent.config.settings import (
    TIKTOKEN_CACHE_DIR,
    OPENAI_EMBEDDING_CONFIG,
    OPENAI_LLM_CONFIG,
    OPENAI_BASE_URL,
)


# Set tiktoken cache directory to avoid downloading it on every run
def setup_cache():
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


setup_cache()


def _llm_gateway_auth_headers() -> dict:
    """Return a Bearer identity token header when running on Cloud Run, empty dict locally."""
    if not os.getenv("K_SERVICE"):
        return {}
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token
        # Audience is the llm-gateway base URL (strip /v1 suffix)
        audience = OPENAI_BASE_URL.rstrip("/")
        if audience.endswith("/v1"):
            audience = audience[:-3]
        token = id_token.fetch_id_token(Request(), audience)
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        return {}


def get_embeddings_model():
    config = {k: v for k, v in OPENAI_EMBEDDING_CONFIG.items() if v}
    config["default_headers"] = _llm_gateway_auth_headers()
    return OpenAIEmbeddings(**config)


def get_llm_model():
    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config["default_headers"] = _llm_gateway_auth_headers()
    return ChatOpenAI(**config)

def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    # Wrap the callback handler in an AsyncCallbackManager
    manager = AsyncCallbackManager(handlers=[callback_handler])

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config.update({"streaming": True, "callbacks": manager, "default_headers": _llm_gateway_auth_headers()})
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
