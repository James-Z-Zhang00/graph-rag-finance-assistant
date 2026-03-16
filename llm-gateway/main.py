"""
LLM Gateway — entry point.

An OpenAI-compatible proxy that sits between internal services and
the upstream LLM provider. Handles auth, request logging, and model
routing transparently.

Run from the llm-gateway/ directory:
    python main.py
    uvicorn main:app --port 8002
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

app = FastAPI(
    title="LLM Gateway",
    description="OpenAI-compatible gateway for routing, auth, and logging.",
    version="1.0.0",
)

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
