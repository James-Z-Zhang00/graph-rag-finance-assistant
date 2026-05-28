"""
Chat router.

POST /chat         — non-streaming search (proxied to search-service)
POST /chat/stream  — streaming search (proxied to search-service)
POST /clear        — clear session (acknowledged; state lives in search-service)
"""

import os
import httpx
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse
from services.kg_service import extract_kg_from_message
from config.settings import SEARCH_SERVICE_URL


def _auth_headers(audience: str) -> dict:
    """Return a Bearer token header when running on Cloud Run, empty dict locally."""
    if not os.getenv("K_SERVICE"):
        return {}
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token
    token = id_token.fetch_id_token(Request(), audience)
    return {"Authorization": f"Bearer {token}"}

logger = logging.getLogger(__name__)
router = APIRouter()

_BASE = SEARCH_SERVICE_URL.rstrip("/")
_SEARCH_URLS = {
    "naive_agent": f"{_BASE}/search/naive",
    "hybrid_agent": f"{_BASE}/search/hybrid",
}
_STREAM_URLS = {
    "naive_agent": f"{_BASE}/search/naive/stream",
    "hybrid_agent": f"{_BASE}/search/hybrid/stream",
}
_CLEAR_URL = f"{_BASE}/search/clear"

_DEFAULT_AGENT = "hybrid_agent"


def _search_url(agent_type: str) -> str:
    return _SEARCH_URLS.get(agent_type, _SEARCH_URLS[_DEFAULT_AGENT])


def _stream_url(agent_type: str) -> str:
    return _STREAM_URLS.get(agent_type, _STREAM_URLS[_DEFAULT_AGENT])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    payload = {
        "query": request.message,
        "session_id": request.session_id,
        "debug": request.debug,
    }
    url = _search_url(request.agent_type)
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=payload, headers=_auth_headers(SEARCH_SERVICE_URL))
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        logger.error("chat proxy error: %s", exc)
        raise HTTPException(status_code=502, detail=f"search-service unreachable: {exc}")

    answer = data.get("answer", "")
    execution_log = data.get("execution_log")

    # Build kg_data from the answer text
    try:
        kg_data = extract_kg_from_message(answer)
    except Exception:
        kg_data = {"nodes": [], "links": []}

    return ChatResponse(
        answer=answer,
        execution_log=execution_log,
        kg_data=kg_data,
        citations=data.get("citations"),
        quality_score=data.get("quality_score"),
        audit_id=data.get("audit_id"),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    payload = {
        "query": request.message,
        "session_id": request.session_id,
        "debug": request.debug,
    }
    url = _stream_url(request.agent_type)

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json=payload, headers=_auth_headers(SEARCH_SERVICE_URL)) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_text():
                        yield chunk
        except Exception as exc:
            logger.error("stream proxy error: %s", exc)
            yield f"[ERROR] {exc}"

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/clear", response_model=ClearResponse)
async def clear(request: ClearRequest):
    # Best-effort: forward clear to search-service if it supports it.
    # If not, just return success — a new session_id will create a fresh agent.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_CLEAR_URL, json={"session_id": request.session_id}, headers=_auth_headers(SEARCH_SERVICE_URL))
            if resp.status_code == 404:
                pass  # search-service has no clear endpoint yet — that's ok
            else:
                resp.raise_for_status()
    except Exception:
        pass  # non-fatal; return success anyway

    return ClearResponse(status="success", remaining_messages="0")
