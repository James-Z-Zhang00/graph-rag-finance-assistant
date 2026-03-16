import os
import json
from pathlib import Path

from dotenv import load_dotenv

# examples and community_algorithm are defined locally so the frontend
# can run as a standalone microservice without the backend package installed.

# Load frontend/.env explicitly so the correct file is always picked up
# regardless of which directory streamlit is launched from.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_FRONTEND_DIR / ".env")


def _get_env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(key: str, default: int) -> int:
    """Read integer environment variable"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} requires an integer value, but got {raw}") from exc


# ===== Frontend Interface and Session Configuration =====

API_URL = os.getenv("FRONTEND_API_URL", "http://localhost:8000/api")  # Backend API address

DEFAULT_AGENT_TYPE = os.getenv("FRONTEND_DEFAULT_AGENT", "hybrid_agent")  # Default Agent
DEFAULT_SHOW_THINKING = _get_env_bool("FRONTEND_SHOW_THINKING", True)  # Default show thinking process
DEFAULT_USE_DEEPER_TOOL = _get_env_bool("FRONTEND_USE_DEEPER_TOOL", True)  # Default deeper tool enabled
DEFAULT_CHAIN_EXPLORATION = _get_env_bool("FRONTEND_USE_CHAIN_EXPLORATION", True)  # Chain exploration enabled

# Community detection algorithm used in KG reasoning (leiden / sllpa).
# Set GRAPH_COMMUNITY_ALGORITHM in the frontend .env to override.
community_algorithm = os.getenv("GRAPH_COMMUNITY_ALGORITHM", "leiden")

# Example questions shown in the sidebar.
# Override by setting FRONTEND_EXAMPLES as a JSON array string in .env, e.g.:
#   FRONTEND_EXAMPLES=["Question one?", "Question two?"]
_examples_env = os.getenv("FRONTEND_EXAMPLES", "")
if _examples_env:
    try:
        examples = json.loads(_examples_env)
    except (json.JSONDecodeError, ValueError):
        examples = []
else:
    examples = [
        "What was Apple's total net sales for fiscal year 2025?",
        "How did JPMorgan's net income change year over year?",
        "What are the main risk factors disclosed in the 10-K?",
        "What were Walmart's comparable sales growth figures?",
    ]

# ===== Knowledge Graph Display Parameters =====

KG_COLOR_PALETTE = [
    "#4285F4",  # Google Blue
    "#EA4335",  # Google Red
    "#FBBC05",  # Google Yellow
    "#34A853",  # Google Green
    "#7B1FA2",  # Purple
    "#0097A7",  # Cyan
    "#FF6D00",  # Orange
    "#757575",  # Gray
    "#607D8B",  # Blue Gray
    "#C2185B"   # Pink
]

NODE_TYPE_COLORS = {
    "Center": "#F0B2F4",     # Center/Source node - Purple
    "Source": "#4285F4",     # Source node - Blue
    "Target": "#EA4335",     # Target node - Red
    "Common": "#34A853",     # Common neighbor - Green
    "Level1": "#0097A7",     # Level 1 association - Cyan
    "Level2": "#FF6D00",     # Level 2 association - Orange
}

DEFAULT_KG_SETTINGS = {
    "physics_enabled": _get_env_bool("KG_PHYSICS_ENABLED", True),
    "node_size": _get_env_int("KG_NODE_SIZE", 25),
    "edge_width": _get_env_int("KG_EDGE_WIDTH", 2),
    "spring_length": _get_env_int("KG_SPRING_LENGTH", 150),
    "gravity": _get_env_int("KG_GRAVITY", -5000),
}
