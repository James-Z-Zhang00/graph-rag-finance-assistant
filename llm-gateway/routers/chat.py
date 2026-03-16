"""
POST /v1/chat/completions

Validates the incoming API key, logs the request, then delegates to the
configured upstream provider. Supports both streaming (SSE) and non-streaming.
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import verify_api_key
from models import ChatCompletionRequest
from providers import provider

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: None = Depends(verify_api_key),
):
    start = time.time()
    request_body = body.model_dump(exclude_none=True)

    logger.info(
        "chat request model=%s stream=%s messages=%d",
        body.model,
        body.stream,
        len(body.messages),
    )

    try:
        if body.stream:
            async def _stream():
                async for chunk in provider.stream_chat(request_body):
                    yield chunk
                logger.info(
                    "chat stream done model=%s elapsed=%.2fs",
                    body.model,
                    time.time() - start,
                )

            return StreamingResponse(_stream(), media_type="text/event-stream")

        result = await provider.chat(request_body)
        logger.info(
            "chat done model=%s elapsed=%.2fs",
            body.model,
            time.time() - start,
        )
        return JSONResponse(content=result)

    except Exception as exc:
        logger.error("chat error model=%s: %s", body.model, exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")
