import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env_int(key: str, default: Optional[int]) -> Optional[int]:
    """Get integer env var, returns default if not set."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {key} requires an integer, got {raw!r}") from exc


def _get_env_float(key: str, default: Optional[float]) -> Optional[float]:
    """Get float env var, returns default if not set."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {key} requires a float, got {raw!r}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean env var, supports true/false/1/0/yes/no."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_choice(key: str, choices: set[str], default: str) -> str:
    """Get a string env var restricted to a fixed set of choices."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(
            f"Env var {key} must be one of {', '.join(sorted(choices))}, got {raw!r}"
        )
    return value


# ===== Base Path Configuration =====

BASE_DIR = Path(__file__).resolve().parent.parent  # graphrag_agent package directory
PROJECT_ROOT = BASE_DIR.parent  # project root
FILES_DIR = PROJECT_ROOT / "files" / "small"
FILE_REGISTRY_PATH = PROJECT_ROOT / "file_registry.json"

# ===== Knowledge Base & System Parameters =====

KB_NAME = "SEC Financial Filings"  # Knowledge base topic, used for deepsearch
workers = _get_env_int("FASTAPI_WORKERS", 2) or 2  # FastAPI concurrent workers

# ===== Knowledge Graph Configuration =====

theme = "SEC Financial Filings Analysis"  # Knowledge graph theme

entity_types = [
    "Company",
    "Filing",
    "FinancialMetric",
    "BusinessSegment",
    "RiskFactor",
    "Executive",
    "Subsidiary",
    "Product",
    "Geography",
    "TimePeriod",
    "Regulation",
]  # Knowledge graph entity types

relationship_types = [
    "REPORTED_IN",
    "HAS_REVENUE",
    "OPERATES_IN",
    "RISK_AFFECTS",
    "MANAGES",
    "OWNS",
    "COMPETES_WITH",
    "FILED_BY",
    "EMPLOYED_BY",
    "SEGMENT_OF",
    "PERIOD_COVERS",
]  # knowledge graph relationship types

# Conflict resolution strategy: manual_first / auto_first / merge
conflict_strategy = os.getenv("GRAPH_CONFLICT_STRATEGY", "manual_first")

# Community detection algorithm: leiden / sllpa
community_algorithm = os.getenv("GRAPH_COMMUNITY_ALGORITHM", "leiden")

# ===== Text Processing Configuration =====

CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 500) or 500  # Text chunk size
OVERLAP = _get_env_int("CHUNK_OVERLAP", 100) or 100  # Chunk overlap length
MAX_TEXT_LENGTH = _get_env_int("MAX_TEXT_LENGTH", 500000) or 500000  # Max text length
similarity_threshold = _get_env_float("SIMILARITY_THRESHOLD", 0.9) or 0.9  # Vector similarity threshold

# ===== Response Generation Configuration =====

response_type = os.getenv("RESPONSE_TYPE", "multiple paragraphs")  # Default response format

# ===== Agent Tool Descriptions =====

lc_description = (
    "For queries requiring specific details. Retrieves specific provisions, clauses, financial figures, "
    "and detailed content from SEC filings. Suitable for questions like 'what was the exact revenue figure' or 'what is the filing process'."
)
gl_description = (
    "For queries requiring summarization. Analyzes the overall framework, financial trends, risk factors, "
    "and business segments across SEC filings. Suitable for questions needing systematic analysis like "
    "'what are the company's main business risks' or 'how does revenue compare across segments'."
)
naive_description = (
    "Basic retrieval tool that directly finds the most relevant text passages for a question without complex analysis. "
    "Quickly retrieves relevant SEC filing content and returns the best matching original text passages."
)


# ===== Performance Tuning Configuration =====

MAX_WORKERS = _get_env_int("MAX_WORKERS", 4) or 4  # Parallel worker threads
BATCH_SIZE = _get_env_int("BATCH_SIZE", 100) or 100  # General batch size
ENTITY_BATCH_SIZE = _get_env_int("ENTITY_BATCH_SIZE", 50) or 50  # Entity batch size
CHUNK_BATCH_SIZE = _get_env_int("CHUNK_BATCH_SIZE", 100) or 100  # Text chunk batch size
EMBEDDING_BATCH_SIZE = _get_env_int("EMBEDDING_BATCH_SIZE", 64) or 64  # Vector batch size
LLM_BATCH_SIZE = _get_env_int("LLM_BATCH_SIZE", 5) or 5  # LLM batch size
COMMUNITY_BATCH_SIZE = _get_env_int("COMMUNITY_BATCH_SIZE", 50) or 50  # Community batch size

# ===== GDS Configuration =====

GDS_MEMORY_LIMIT = _get_env_int("GDS_MEMORY_LIMIT", 6) or 6  # GB
GDS_CONCURRENCY = _get_env_int("GDS_CONCURRENCY", 4) or 4
GDS_NODE_COUNT_LIMIT = _get_env_int("GDS_NODE_COUNT_LIMIT", 50000) or 50000
GDS_TIMEOUT_SECONDS = _get_env_int("GDS_TIMEOUT_SECONDS", 300) or 300

# ===== Entity Disambiguation & Alignment Configuration =====

DISAMBIG_STRING_THRESHOLD = _get_env_float("DISAMBIG_STRING_THRESHOLD", 0.7) or 0.7
DISAMBIG_VECTOR_THRESHOLD = _get_env_float("DISAMBIG_VECTOR_THRESHOLD", 0.85) or 0.85
DISAMBIG_NIL_THRESHOLD = _get_env_float("DISAMBIG_NIL_THRESHOLD", 0.6) or 0.6
DISAMBIG_TOP_K = _get_env_int("DISAMBIG_TOP_K", 5) or 5

ALIGNMENT_CONFLICT_THRESHOLD = (
    _get_env_float("ALIGNMENT_CONFLICT_THRESHOLD", 0.5) or 0.5
)
ALIGNMENT_MIN_GROUP_SIZE = _get_env_int("ALIGNMENT_MIN_GROUP_SIZE", 2) or 2

# ===== Path & Cache Configuration =====

DEFAULT_CACHE_ROOT = Path(
    os.getenv("CACHE_ROOT", PROJECT_ROOT / "cache")
).expanduser()
MODEL_CACHE_ROOT = Path(
    os.getenv("MODEL_CACHE_ROOT", DEFAULT_CACHE_ROOT)
).expanduser()
MODEL_CACHE_DIR = MODEL_CACHE_ROOT / "model"
CACHE_DIR = Path(os.getenv("CACHE_DIR", DEFAULT_CACHE_ROOT)).expanduser()
TIKTOKEN_CACHE_DIR = Path(
    os.getenv("TIKTOKEN_CACHE_DIR", DEFAULT_CACHE_ROOT / "tiktoken")
).expanduser()
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

SENTENCE_TRANSFORMER_MODELS = [
    item.strip()
    for item in os.getenv("SENTENCE_TRANSFORMER_MODELS", "").split(",")
    if item.strip()
]  # preloaded local models
CACHE_EMBEDDING_PROVIDER = os.getenv(
    "CACHE_EMBEDDING_PROVIDER", "sentence_transformer"
).lower()
CACHE_SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "CACHE_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)

CACHE_SETTINGS = {
    "dir": CACHE_DIR,
    "memory_only": _get_env_bool("CACHE_MEMORY_ONLY", False),
    "max_memory_size": _get_env_int("CACHE_MAX_MEMORY_SIZE", 100) or 100,
    "max_disk_size": _get_env_int("CACHE_MAX_DISK_SIZE", 1000) or 1000,
    "thread_safe": _get_env_bool("CACHE_THREAD_SAFE", True),
    "enable_vector_similarity": _get_env_bool(
        "CACHE_ENABLE_VECTOR_SIMILARITY", True
    ),
    "similarity_threshold": _get_env_float(
        "CACHE_SIMILARITY_THRESHOLD", similarity_threshold
    )
    or similarity_threshold,
    "max_vectors": _get_env_int("CACHE_MAX_VECTORS", 10000) or 10000,
}

# ===== Neo4j Connection Configuration =====

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_MAX_POOL_SIZE = _get_env_int("NEO4J_MAX_POOL_SIZE", 10) or 10
NEO4J_REFRESH_SCHEMA = _get_env_bool("NEO4J_REFRESH_SCHEMA", False)

NEO4J_CONFIG = {
    "uri": NEO4J_URI,
    "username": NEO4J_USERNAME,
    "password": NEO4J_PASSWORD,
    "max_pool_size": NEO4J_MAX_POOL_SIZE,
    "refresh_schema": NEO4J_REFRESH_SCHEMA,
}

# ===== LLM & Embedding Model Configuration =====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL") or None
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL") or None
LLM_TEMPERATURE = _get_env_float("TEMPERATURE", None)
LLM_MAX_TOKENS = _get_env_int("MAX_TOKENS", None)

OPENAI_EMBEDDING_CONFIG = {
    "model": OPENAI_EMBEDDINGS_MODEL,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

OPENAI_LLM_CONFIG = {
    "model": OPENAI_LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

# ===== Similar Entity Detection Parameters =====

SIMILAR_ENTITY_SETTINGS = {
    "word_edit_distance": _get_env_int("SIMILAR_ENTITY_WORD_EDIT_DISTANCE", 3) or 3,
    "batch_size": _get_env_int("SIMILAR_ENTITY_BATCH_SIZE", 500) or 500,
    "memory_limit": _get_env_int(
        "SIMILAR_ENTITY_MEMORY_LIMIT", GDS_MEMORY_LIMIT
    )
    or GDS_MEMORY_LIMIT,
    "top_k": _get_env_int("SIMILAR_ENTITY_TOP_K", 10) or 10,
}

# ===== Search Tool Configuration =====

BASE_SEARCH_CONFIG = {
    "cache_max_size": _get_env_int("SEARCH_CACHE_MEMORY_SIZE", 200) or 200,
    "vector_limit": _get_env_int("SEARCH_VECTOR_LIMIT", 5) or 5,
    "text_limit": _get_env_int("SEARCH_TEXT_LIMIT", 5) or 5,
    "semantic_top_k": _get_env_int("SEARCH_SEMANTIC_TOP_K", 5) or 5,
    "relevance_top_k": _get_env_int("SEARCH_RELEVANCE_TOP_K", 5) or 5,
}

LOCAL_SEARCH_SETTINGS = {
    "top_chunks": _get_env_int("LOCAL_SEARCH_TOP_CHUNKS", 3) or 3,
    "top_communities": _get_env_int("LOCAL_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "top_outside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_OUTSIDE_RELS", 10
    )
    or 10,
    "top_inside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_INSIDE_RELS", 10
    )
    or 10,
    "top_entities": _get_env_int("LOCAL_SEARCH_TOP_ENTITIES", 10) or 10,
    "index_name": os.getenv("LOCAL_SEARCH_INDEX_NAME", "vector"),
}

GLOBAL_SEARCH_SETTINGS = {
    "default_level": _get_env_int("GLOBAL_SEARCH_LEVEL", 0) or 0,
    "community_batch_size": _get_env_int("GLOBAL_SEARCH_BATCH_SIZE", 5) or 5,
}

NAIVE_SEARCH_TOP_K = _get_env_int("NAIVE_SEARCH_TOP_K", 3) or 3

HYBRID_SEARCH_SETTINGS = {
    "entity_limit": _get_env_int("HYBRID_SEARCH_ENTITY_LIMIT", 15) or 15,
    "max_hop_distance": _get_env_int("HYBRID_SEARCH_MAX_HOP", 2) or 2,
    "top_communities": _get_env_int("HYBRID_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "batch_size": _get_env_int("HYBRID_SEARCH_BATCH_SIZE", 10) or 10,
    "community_level": _get_env_int("HYBRID_SEARCH_COMMUNITY_LEVEL", 0) or 0,
}

# ===== Agent Configuration =====

AGENT_SETTINGS = {
    "default_recursion_limit": _get_env_int("AGENT_RECURSION_LIMIT", 5) or 5,
    "chunk_size": _get_env_int("AGENT_CHUNK_SIZE", 4) or 4,
    "stream_flush_threshold": _get_env_int("AGENT_STREAM_FLUSH_THRESHOLD", 40)
    or 40,
    "deep_stream_flush_threshold": _get_env_int(
        "DEEP_AGENT_STREAM_FLUSH_THRESHOLD", 80
    )
    or 80,
    "fusion_stream_flush_threshold": _get_env_int(
        "FUSION_AGENT_STREAM_FLUSH_THRESHOLD", 60
    )
    or 60,
}

# ===== Multi-Agent (Plan-Execute-Report) Configuration =====

MULTI_AGENT_PLANNER_MAX_TASKS = _get_env_int("MA_PLANNER_MAX_TASKS", 6) or 6
MULTI_AGENT_ALLOW_UNCLARIFIED_PLAN = _get_env_bool("MA_ALLOW_UNCLARIFIED_PLAN", True)
MULTI_AGENT_DEFAULT_DOMAIN = os.getenv("MA_DEFAULT_DOMAIN", "general")

MULTI_AGENT_AUTO_GENERATE_REPORT = _get_env_bool("MA_AUTO_GENERATE_REPORT", True)
MULTI_AGENT_STOP_ON_CLARIFICATION = _get_env_bool("MA_STOP_ON_CLARIFICATION", True)
MULTI_AGENT_STRICT_PLAN_SIGNAL = _get_env_bool("MA_STRICT_PLAN_SIGNAL", True)

MULTI_AGENT_DEFAULT_REPORT_TYPE = os.getenv("MA_DEFAULT_REPORT_TYPE", "long_document")
MULTI_AGENT_ENABLE_CONSISTENCY_CHECK = _get_env_bool(
    "MA_ENABLE_CONSISTENCY_CHECK", True
)
MULTI_AGENT_ENABLE_MAPREDUCE = _get_env_bool("MA_ENABLE_MAPREDUCE", True)
MULTI_AGENT_MAPREDUCE_THRESHOLD = _get_env_int("MA_MAPREDUCE_THRESHOLD", 20) or 20
MULTI_AGENT_MAX_TOKENS_PER_REDUCE = (
    _get_env_int("MA_MAX_TOKENS_PER_REDUCE", 4000) or 4000
)
MULTI_AGENT_ENABLE_PARALLEL_MAP = _get_env_bool("MA_ENABLE_PARALLEL_MAP", True)

MULTI_AGENT_SECTION_MAX_EVIDENCE = (
    _get_env_int("MA_SECTION_MAX_EVIDENCE", 8) or 8
)
MULTI_AGENT_SECTION_MAX_CONTEXT_CHARS = (
    _get_env_int("MA_SECTION_MAX_CONTEXT_CHARS", 800) or 800
)
MULTI_AGENT_REFLECTION_ALLOW_RETRY = _get_env_bool(
    "MA_REFLECTION_ALLOW_RETRY", False
)
MULTI_AGENT_REFLECTION_MAX_RETRIES = (
    _get_env_int("MA_REFLECTION_MAX_RETRIES", 1) or 1
)
MULTI_AGENT_WORKER_EXECUTION_MODE = _get_env_choice(
    "MA_WORKER_EXECUTION_MODE",
    {"sequential", "parallel"},
    "sequential",
)
MULTI_AGENT_WORKER_MAX_CONCURRENCY = (
    _get_env_int("MA_WORKER_MAX_CONCURRENCY", MAX_WORKERS) or MAX_WORKERS
)
if MULTI_AGENT_WORKER_MAX_CONCURRENCY < 1:
    raise ValueError("MA_WORKER_MAX_CONCURRENCY must be >= 1")
