"""
Cache management routes.

POST /search/clear  — clear in-memory cache (specific session or all sessions)
"""
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from hybrid_search_agent.agent import agent_pool

logger = logging.getLogger(__name__)
router = APIRouter()


class ClearRequest(BaseModel):
    session_id: Optional[str] = None


class ClearResponse(BaseModel):
    status: str
    cleared: str


@router.post("/search/clear", response_model=ClearResponse)
async def clear_cache(request: ClearRequest = ClearRequest()) -> ClearResponse:
    """Clear in-memory cache for a specific session or all sessions."""
    if request.session_id:
        with agent_pool._lock:
            agent = agent_pool._instances.get(request.session_id)

        if agent:
            agent.cache_manager.clear()
            agent.global_cache_manager.clear()
            cleared = f"session:{request.session_id}"
        else:
            cleared = "none (session not found)"
    else:
        with agent_pool._lock:
            agents = list(agent_pool._instances.values())
            session_count = len(agents)

        for agent in agents:
            agent.cache_manager.clear()
            agent.global_cache_manager.clear()

        cleared = f"all ({session_count} sessions)"

    logger.info("Cache cleared: %s", cleared)
    return ClearResponse(status="success", cleared=cleared)
