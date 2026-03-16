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

HOST = os.getenv("SEARCH_SERVICE_HOST", "0.0.0.0")
PORT = _get_env_int("SEARCH_SERVICE_PORT", 8003)
RELOAD = _get_env_bool("SEARCH_SERVICE_RELOAD", False)
LOG_LEVEL = os.getenv("SEARCH_SERVICE_LOG_LEVEL", "info")

# ===== Uvicorn Config =====

UVICORN_CONFIG = {
    "host": HOST,
    "port": PORT,
    "reload": RELOAD,
    "log_level": LOG_LEVEL,
    "workers": 1,
}
