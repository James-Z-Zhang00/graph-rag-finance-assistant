"""
GraphRAG runner — calls the deployed search-service /search/hybrid endpoint.

Returns: {"answer": str, "contexts": List[str], "quality_score": float}
"""

import requests
from typing import Optional
from config import SEARCH_SERVICE_URL, SEARCH_SERVICE_TOKEN


def _headers() -> dict:
    if SEARCH_SERVICE_TOKEN:
        return {"Authorization": f"Bearer {SEARCH_SERVICE_TOKEN}"}
    return {}


def query(question: str, session_id: str = "eval") -> dict:
    """
    Send a question to the deployed GraphRAG search service.

    Returns a dict with:
      answer        — generated answer string
      contexts      — list of retrieved context strings (from graph + vector retrieval)
      quality_score — faithfulness score from internal HallucinationValidator (0–1)
    """
    payload = {"query": question, "session_id": session_id, "debug": False}
    response = requests.post(
        f"{SEARCH_SERVICE_URL}/search/hybrid",
        json=payload,
        headers=_headers(),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("answer", ""),
        "contexts": data.get("contexts") or [],
        "quality_score": data.get("quality_score"),
    }
