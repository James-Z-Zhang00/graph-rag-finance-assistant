"""
Build service entry point.
Port: 8004
"""

import sys
import os

# Add service root to sys.path so internal imports work (routers, build_pipeline, config, graphrag_agent)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from routers.health import router as health_router
from routers.build import router as build_router, _executor
from routers.files import router as files_router
from config.settings import BUILD_SERVICE_HOST, BUILD_SERVICE_PORT, BUILD_SERVICE_RELOAD, BUILD_SERVICE_LOG_LEVEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # On shutdown (SIGTERM from Cloud Run), wait for any running build thread to finish
    # before the container exits so we don't lose a build mid-flight.
    print("[main] shutdown: waiting for background build thread to finish...")
    _executor.shutdown(wait=True, cancel_futures=False)
    print("[main] shutdown: build thread done")


app = FastAPI(title="Build Service", description="Graph build pipeline microservice", lifespan=lifespan)

app.include_router(health_router)
app.include_router(build_router)
app.include_router(files_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=BUILD_SERVICE_HOST,
        port=BUILD_SERVICE_PORT,
        reload=BUILD_SERVICE_RELOAD,
        log_level=BUILD_SERVICE_LOG_LEVEL,
        workers=1,  # Must be 1 — build jobs use in-process ThreadPoolExecutor
    )
