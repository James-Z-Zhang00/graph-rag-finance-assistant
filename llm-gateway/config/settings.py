import os
from dotenv import load_dotenv

load_dotenv()


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {key} requires int, got {raw!r}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


# ===== Service Network Config =====

HOST = os.getenv("LLM_GATEWAY_HOST", "0.0.0.0")
PORT = _get_env_int("LLM_GATEWAY_PORT", 8002)
RELOAD = _get_env_bool("LLM_GATEWAY_RELOAD", False)
LOG_LEVEL = os.getenv("LLM_GATEWAY_LOG_LEVEL", "info")

# ===== Auth =====

# Clients (e.g. graphrag_agent) must send this key as Bearer token.
# Set to empty string to disable auth (dev only).
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()

# ===== Upstream Provider =====

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "").strip()

# ===== Request Config =====

# Seconds before giving up on the upstream (streaming keeps the connection alive)
REQUEST_TIMEOUT = _get_env_int("REQUEST_TIMEOUT", 120)

# ===== Uvicorn Config =====

UVICORN_CONFIG = {
    "host": HOST,
    "port": PORT,
    "reload": RELOAD,
    "log_level": LOG_LEVEL,
    "workers": 1,
}
