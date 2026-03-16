"""
Pipeline runner — executes graph build stages in a background thread.
Each stage updates the job's stage field so callers can poll progress.
"""

import os
import multiprocessing as mp
import traceback
from pathlib import Path


def _auth_headers(audience: str) -> dict:
    """Return a Bearer token header when running on Cloud Run, empty dict locally."""
    if not os.getenv("K_SERVICE"):
        return {}
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token
    token = id_token.fetch_id_token(Request(), audience)
    return {"Authorization": f"Bearer {token}"}

# Must be called before any multiprocessing-spawning code is imported
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

from build_pipeline.job_store import job_store


def run_full_build(job_id: str, sec_files_dir: str, sec_parser_url: str,
                   gcs_bucket: str = "", gcs_prefix: str = "small/"):
    """
    Full build pipeline:
      1. (Optional) Download source files from GCS to a temp dir
      2. Call sec-parser to parse files → structured JSON
      3. Drop all Neo4j indexes
      4. Build base graph from parsed documents (skips file reading)
      5. Build entity indexes + community detection
      6. Build chunk index
    """
    import tempfile
    try:
        import requests

        # Stage 1 (optional): pull files from GCS into a temp dir
        if gcs_bucket:
            job_store.mark_running(job_id, stage="gcs_download")
            tmp_dir = tempfile.mkdtemp(prefix="gcs-sec-files-")
            from services.gcs_storage import download_prefix_to_dir
            n = download_prefix_to_dir(gcs_bucket, gcs_prefix, tmp_dir)
            print(f"[runner] downloaded {n} file(s) from gs://{gcs_bucket}/{gcs_prefix}")
            sec_files_dir = tmp_dir

        # Stage 2: upload files to sec-parser and get back parsed documents
        job_store.mark_running(job_id, stage="sec_parse")
        parse_url = sec_parser_url.rstrip("/") + "/api/sec/parse"
        file_paths = [
            p for p in Path(sec_files_dir).rglob("*") if p.is_file()
        ]
        upload_files = []
        open_handles = []
        for p in file_paths:
            fh = open(p, "rb")
            open_handles.append(fh)
            upload_files.append(("files", (p.name, fh, "application/octet-stream")))
        try:
            resp = requests.post(
                parse_url,
                files=upload_files,
                headers=_auth_headers(sec_parser_url),
                timeout=600,
            )
        finally:
            for fh in open_handles:
                fh.close()
        resp.raise_for_status()
        documents = resp.json()["documents"]
        print(f"[runner] sec-parser returned {len(documents)} document(s)")

        # Lazy imports after mp.set_start_method
        from graphrag_agent.graph.core import connection_manager
        from graphrag_agent.integrations.build.build_graph import KnowledgeGraphBuilder
        from graphrag_agent.integrations.build.build_index_and_community import IndexCommunityBuilder
        from graphrag_agent.integrations.build.build_chunk_index import ChunkIndexBuilder

        # Stage 2: drop indexes
        job_store.update(job_id, stage="drop_indexes")
        connection_manager.drop_all_indexes()

        # Stage 3: build base graph from parsed documents
        job_store.update(job_id, stage="build_graph")
        graph_builder = KnowledgeGraphBuilder()
        graph_builder.build_from_documents(documents)

        # Stage 4: index entities + community detection
        job_store.update(job_id, stage="index_community")
        index_builder = IndexCommunityBuilder()
        index_builder.process()

        # Stage 5: chunk index
        job_store.update(job_id, stage="chunk_index")
        chunk_builder = ChunkIndexBuilder()
        chunk_builder.process()

        job_store.mark_completed(job_id, stats={
            "documents_parsed": len(documents),
            "stages_completed": 5,
        })

    except Exception as exc:
        job_store.mark_failed(job_id, error=traceback.format_exc())


def run_incremental_build(job_id: str, files_dir: str, registry_path: str):
    """Execute incremental update pipeline."""
    try:
        job_store.mark_running(job_id, stage="detect_changes")

        from graphrag_agent.integrations.build.incremental_graph_builder import IncrementalGraphUpdater

        updater = IncrementalGraphUpdater(files_dir=files_dir, registry_path=registry_path)

        job_store.update(job_id, stage="incremental_update")
        stats = updater.process_incremental_update()

        job_store.mark_completed(job_id, stats={
            "files_processed": stats.get("files_processed", 0),
            "entities_integrated": stats.get("entities_integrated", 0),
            "relations_integrated": stats.get("relations_integrated", 0),
            "total_time": stats.get("total_time", 0),
        })

    except Exception as exc:
        job_store.mark_failed(job_id, error=traceback.format_exc())
