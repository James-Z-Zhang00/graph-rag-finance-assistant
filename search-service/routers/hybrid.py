"""
POST /search/hybrid         — non-streaming search
POST /search/hybrid/stream  — streaming search (plain text chunks)
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hybrid_search_agent.agent import agent_pool
from models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: SearchRequest):
    agent = agent_pool.get(request.session_id)
    try:
        if request.debug:
            result = await asyncio.to_thread(
                agent.ask_with_trace, request.query, thread_id=request.session_id
            )
            return SearchResponse(
                answer=result["answer"],
                execution_log=result.get("execution_log"),
            )
        else:
            answer = await asyncio.to_thread(
                agent.ask, request.query, thread_id=request.session_id
            )
            return SearchResponse(answer=answer)
    except Exception as exc:
        logger.error("hybrid search error session=%s: %s", request.session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/search/hybrid/stream")
async def search_hybrid_stream(request: SearchRequest):
    agent = agent_pool.get(request.session_id)

    async def generate():
        try:
            async for chunk in agent.ask_stream(request.query, thread_id=request.session_id):
                yield chunk
        except Exception as exc:
            logger.error("hybrid stream error session=%s: %s", request.session_id, exc)
            yield f"[ERROR] {exc}"

    return StreamingResponse(generate(), media_type="text/plain")
