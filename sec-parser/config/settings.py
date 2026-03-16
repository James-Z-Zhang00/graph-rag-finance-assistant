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
        raise ValueError(f"Env var {key} requires int, got {raw}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


# ===== Service Network Config =====

BUILD_SERVICE_HOST = os.getenv("SEC_PARSER_HOST", "0.0.0.0")
BUILD_SERVICE_PORT = _get_env_int("SEC_PARSER_PORT", 8001)
BUILD_SERVICE_RELOAD = _get_env_bool("SEC_PARSER_RELOAD", False)
BUILD_SERVICE_LOG_LEVEL = os.getenv("SEC_PARSER_LOG_LEVEL", "info")

# ===== SEC / Chunking Config =====

CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 500) or 500
OVERLAP = _get_env_int("CHUNK_OVERLAP", 100) or 100
MAX_TEXT_LENGTH = _get_env_int("MAX_TEXT_LENGTH", 500000) or 500000

# ===== Downstream Ingest Config =====

INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/api/ingest/sec")

# ===== Uvicorn Config =====

UVICORN_CONFIG = {
    "host": BUILD_SERVICE_HOST,
    "port": BUILD_SERVICE_PORT,
    "reload": BUILD_SERVICE_RELOAD,
    "log_level": BUILD_SERVICE_LOG_LEVEL,
    "workers": 1,
}
