"""
Audit Logger — writes structured JSONL audit events per query lifecycle.

Each query produces events: query_received, retrieval_done, generation_done.
PII masking events are also recorded (type counts only, never raw PII values).

Log files are written to /tmp/audit_logs/audit_YYYY-MM-DD.jsonl (daily rotation).
"""

import json
import uuid
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit trail writer."""

    def __init__(self, log_dir: str = "/tmp/audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        return self.log_dir / f"audit_{date.today().isoformat()}.jsonl"

    def log(self, event_type: str, session_id: str, details: Dict[str, Any]) -> str:
        """
        Append a structured audit event to today's log file.

        Args:
            event_type: One of query_received | retrieval_done | generation_done
                        | pii_masked | validation_done.
            session_id: The session/thread identifier for this request.
            details:    Arbitrary key-value metadata (must be JSON-serialisable).
                        MUST NOT contain raw PII values.

        Returns:
            event_id (UUID4 string) for cross-referencing.
        """
        event_id = str(uuid.uuid4())
        entry = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "session_id": session_id,
            "details": details,
        }
        try:
            with open(self._log_path(), "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("audit write failed event_type=%s: %s", event_type, e)
        return event_id
