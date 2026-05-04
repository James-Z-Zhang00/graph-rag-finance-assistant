"""
Build job router.

GET  /build/check        — connectivity check (Neo4j, sec-parser, llm-gateway)
POST /build/full         — trigger full 4-stage pipeline
POST /build/incremental  — trigger incremental update
GET  /build/jobs         — list all jobs
GET  /build/jobs/{id}    — get single job status
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException

from models import TriggerResponse, JobResponse
from build_pipeline.job_store import job_store
from build_pipeline.runner import run_full_build, run_incremental_build, _auth_headers
from config.settings import FILES_DIR, FILE_REGISTRY_PATH, SEC_PARSER_URL, SEC_FILES_DIR, GCS_BUCKET_NAME, GCS_FILES_PREFIX

router = APIRouter(prefix="/build")

# Single-worker executor so builds run one at a time
_executor = ThreadPoolExecutor(max_workers=1)


@router.get("/check")
def connectivity_check():
    """
    Test connectivity to Neo4j, sec-parser, and llm-gateway.
    Returns a status dict for each — no data is read or written.
    """
    import requests
    from graphrag_agent.config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_BASE_URL

    results = {}

    # --- Neo4j ---
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        results["neo4j"] = {"ok": True, "uri": NEO4J_URI}
    except Exception as e:
        results["neo4j"] = {"ok": False, "uri": NEO4J_URI, "error": str(e)}

    # --- sec-parser ---
    try:
        url = SEC_PARSER_URL.rstrip("/") + "/health"
        resp = requests.get(url, headers=_auth_headers(SEC_PARSER_URL), timeout=10)
        results["sec_parser"] = {"ok": resp.status_code == 200, "url": url, "status_code": resp.status_code}
    except Exception as e:
        results["sec_parser"] = {"ok": False, "url": SEC_PARSER_URL, "error": str(e)}

    # --- llm-gateway ---
    try:
        # OPENAI_BASE_URL is like https://llm-gateway.../v1 — health is at the root
        base = OPENAI_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = base + "/health"
        resp = requests.get(url, headers=_auth_headers(base), timeout=10)
        results["llm_gateway"] = {"ok": resp.status_code == 200, "url": url, "status_code": resp.status_code}
    except Exception as e:
        results["llm_gateway"] = {"ok": False, "url": OPENAI_BASE_URL, "error": str(e)}

    results["all_ok"] = all(v["ok"] for v in results.values() if isinstance(v, dict))
    return results


def _log_build_done(future):
    """Callback that logs any unhandled exception that escaped the build runner."""
    exc = future.exception()
    if exc:
        import traceback as _tb
        print(f"[build] unhandled exception in build thread:\n{''.join(_tb.format_exception(type(exc), exc, exc.__traceback__))}")


@router.post("/full", response_model=TriggerResponse)
async def trigger_full_build():
    """Trigger a full graph build (drop indexes → build graph → index community → chunk index)."""
    # Reject if a build is already running
    running = [j for j in job_store.list_all() if j["status"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail=f"A build is already running: {running[0]['job_id']}")

    job = job_store.create("full")
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_executor, run_full_build, job.job_id, SEC_FILES_DIR, SEC_PARSER_URL, GCS_BUCKET_NAME, GCS_FILES_PREFIX)
    future.add_done_callback(_log_build_done)
    return TriggerResponse(job_id=job.job_id, message="Full build started")


@router.post("/incremental", response_model=TriggerResponse)
async def trigger_incremental_build():
    """Trigger an incremental update (detects changed files and updates graph)."""
    running = [j for j in job_store.list_all() if j["status"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail=f"A build is already running: {running[0]['job_id']}")

    job = job_store.create("incremental")
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_executor, run_incremental_build, job.job_id, FILES_DIR, FILE_REGISTRY_PATH)
    future.add_done_callback(_log_build_done)
    return TriggerResponse(job_id=job.job_id, message="Incremental build started")


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs():
    """List all build jobs."""
    return job_store.list_all()


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get status of a specific job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
