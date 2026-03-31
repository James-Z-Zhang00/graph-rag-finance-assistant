"""
Audit Logger — writes structured audit events per query lifecycle.

Backends
--------
stdout (default / Cloud Run)
    Each event is emitted as a single JSON line to stdout.
    Cloud Run automatically ships stdout to Cloud Logging — no disk usage,
    no container bloat, queryable and retentionable via GCP log policies.

file (local development)
    Appends JSONL to <AUDIT_LOG_DIR>/audit_YYYY-MM-DD.jsonl (daily rotation).

Controlled by AUDIT_LOG_BACKEND env var ("stdout" | "file"), default "stdout".

Schema (every event)
--------------------
{
    "event_id":       "<uuid4>",
    "timestamp":      "<ISO8601Z>",
    "event_type":     "query_received | pii_masked | retrieval_done |
                        generation_done | validation_done |
                        output_pii_detected | dlp_scan_error | dlp_timeout",
    "session_id":     "<thread_id>",
    "request_id":     "<uuid4 | null>",   -- groups all events for one request
    "pipeline_stage": "<string | null>",  -- input | input_masking | retrieval |
                                             generation | hallucination_validation |
                                             output_scanning
    "details":        { ... }             -- event-specific payload, no raw PII
}
"""

import json
import sys
import uuid
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Structured audit event writer supporting stdout and file backends."""

    def __init__(self, log_dir: Optional[str] = None, backend: Optional[str] = None):
        try:
            from graphrag_agent.config.settings import COMPLIANCE_SETTINGS
            self._backend = backend or COMPLIANCE_SETTINGS["audit_log_backend"]
            self._log_dir = Path(log_dir or COMPLIANCE_SETTINGS["audit_log_dir"])
        except Exception:
            self._backend = backend or "stdout"
            self._log_dir = Path(log_dir or "/tmp/audit_logs")

        if self._backend == "file":
            self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        session_id: str,
        details: Dict[str, Any],
        pipeline_stage: Optional[str] = None,
    ) -> str:
        """
        Emit a structured audit event.

        Parameters
        ----------
        event_type:     Registered event type (see module docstring).
        session_id:     Session/thread identifier for this request.
        details:        Key-value metadata. MUST NOT contain raw PII values.
        pipeline_stage: Pipeline phase this event belongs to.

        Returns
        -------
        event_id (UUID4 string) for cross-referencing.
        """
        try:
            from graphrag_agent.compliance.request_context import get_request_id
            request_id = get_request_id()
        except Exception:
            request_id = None

        event_id = str(uuid.uuid4())
        entry = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "session_id": session_id,
            "request_id": request_id,
            "pipeline_stage": pipeline_stage,
            "details": details,
        }

        if self._backend == "stdout":
            self._emit_stdout(entry)
        else:
            self._emit_file(entry)

        return event_id

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_stdout(entry: Dict[str, Any]) -> None:
        """Write one JSON line to stdout — picked up by Cloud Run → Cloud Logging."""
        try:
            print(json.dumps(entry), flush=True)
        except Exception as e:
            logger.error("audit stdout write failed event_type=%s: %s", entry.get("event_type"), e)

    def _emit_file(self, entry: Dict[str, Any]) -> None:
        """Append one JSON line to today's JSONL file."""
        log_path = self._log_dir / f"audit_{date.today().isoformat()}.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("audit file write failed event_type=%s: %s", entry.get("event_type"), e)
