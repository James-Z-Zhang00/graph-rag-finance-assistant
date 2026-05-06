"""
POST /search/hybrid         — non-streaming search
POST /search/hybrid/stream  — streaming search (plain text chunks)
"""

import asyncio
import json
import logging
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hybrid_search_agent.agent import agent_pool
from models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _gcp_log(severity: str, **fields):
    print(json.dumps({"severity": severity, **fields}), flush=True, file=sys.stdout)


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: SearchRequest):
    agent = agent_pool.get(request.session_id)
    try:
        # Always use ask_with_trace so PII masking and audit logging run on every request.
        # execution_log is only included in the response when debug=True.
        result = await asyncio.to_thread(
            agent.ask_with_trace, request.query, thread_id=request.session_id
        )
        response = SearchResponse(
            answer=result["answer"],
            execution_log=result.get("execution_log") if request.debug else None,
            citations=result.get("citations"),
            quality_score=result.get("quality_score"),
            audit_id=result.get("audit_id"),
            contexts=result.get("contexts"),
        )
        _gcp_log("INFO",
            message="search response",
            session_id=request.session_id,
            query=request.query,
            quality_score=response.quality_score,
            answer=response.answer,
        )
        return response
    except Exception as exc:
        _gcp_log("ERROR",
            message="hybrid search error",
            session_id=request.session_id,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/search/hybrid/stream")
async def search_hybrid_stream(request: SearchRequest):
    agent = agent_pool.get(request.session_id)

    async def generate():
        try:
            # Use ask_stream_with_compliance so PII masking, audit logging,
            # and DLP output scanning run on every streaming request.
            async for chunk in agent.ask_stream_with_compliance(
                request.query, thread_id=request.session_id
            ):
                yield chunk
        except Exception as exc:
            logger.error("hybrid stream error session=%s: %s", request.session_id, exc)
            yield f"[ERROR] {exc}"

    return StreamingResponse(generate(), media_type="text/plain")
