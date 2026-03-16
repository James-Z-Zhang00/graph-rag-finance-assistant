"""
Search Service — entry point.

Hosts search agents as standalone microservices. Currently supports hybrid search.
Additional agents (naive_rag, graph) can be added as new routers.

Run from the search-service/ directory:
    python main.py
    uvicorn main:app --port 8003
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uvicorn
from fastapi import FastAPI
from routers import api_router
from config.settings import UVICORN_CONFIG
from hybrid_search_agent.agent import agent_pool

app = FastAPI(
    title="Search Service",
    description="Knowledge graph search agents — hybrid, naive RAG, graph (planned).",
    version="1.0.0",
)

app.include_router(api_router)


@app.on_event("shutdown")
def shutdown_event():
    agent_pool.close_all()


if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
