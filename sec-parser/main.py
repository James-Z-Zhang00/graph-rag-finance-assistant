"""
SEC File Parser Service — entry point.

A lightweight standalone microservice that reads SEC filings from a local
directory, parses them through the SEC pipeline, and returns structured JSON.

Run from the sec-parser/ directory:
    python main.py
    uvicorn main:app --port 8001

All internal modules use unqualified imports (e.g. `from routers import ...`)
because this directory is added to sys.path at startup.
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
    title="SEC File Parser Service",
    description="Parses SEC filings (PDF, HTML, XBRL, TXT) and returns structured JSON for LLM and graph writer.",
    version="2.0.0",
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
