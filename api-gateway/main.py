"""
API Gateway — entry point.

Serves the frontend at http://localhost:8000/api.
Routes:
  /api/chat, /api/chat/stream  → proxied to search-service :8003
  /api/clear                   → session clear (best-effort)
  /api/knowledge_graph/*       → direct Neo4j
  /api/entity_types, /api/relation_types
  /api/entities/search, /api/relations/search
  /api/entity/create|update|delete
  /api/relation/create|update|delete
  /api/kg_reasoning
  /api/source, /api/source_info, /api/content_batch, /api/source_info_batch
  /api/feedback

Run from the api-gateway/ directory:
    python main.py
    uvicorn main:app --port 8000
"""

# Only the chat API connects the front and the back end, others are just how frontend draw graphs for the UI

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import api_router
from config.settings import GATEWAY_HOST, GATEWAY_PORT, GATEWAY_RELOAD, GATEWAY_LOG_LEVEL

app = FastAPI(
    title="API Gateway",
    description="Single entry point for the GraphRAG frontend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        reload=GATEWAY_RELOAD,
        log_level=GATEWAY_LOG_LEVEL,
        workers=1,
    )
