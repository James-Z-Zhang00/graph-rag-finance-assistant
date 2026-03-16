"""
POST /v1/embeddings

Validates the incoming API key, logs the request, then delegates to the
configured upstream provider.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from auth import verify_api_key
from models import EmbeddingRequest
from providers import provider

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embeddings")
async def embeddings(
    body: EmbeddingRequest,
    _: None = Depends(verify_api_key),
):
    start = time.time()
    request_body = body.model_dump(exclude_none=True)

    input_preview = body.input if isinstance(body.input, str) else f"[list len={len(body.input)}]"
    logger.info("embeddings request model=%s input=%s", body.model, input_preview[:80])

    try:
        result = await provider.embeddings(request_body)
        logger.info(
            "embeddings done model=%s elapsed=%.2fs",
            body.model,
            time.time() - start,
        )
        return JSONResponse(content=result)

    except Exception as exc:
        logger.error("embeddings error model=%s: %s", body.model, exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")
