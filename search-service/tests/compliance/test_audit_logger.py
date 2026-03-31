"""
Unit tests for AuditLogger — schema validation and failure handling.
"""

import json
import pytest

from graphrag_agent.compliance.audit_logger import AuditLogger
from graphrag_agent.compliance.request_context import set_request_id, clear_request_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logger(audit_log_dir) -> AuditLogger:
    """File-backend logger for tests that inspect written JSONL."""
    return AuditLogger(log_dir=str(audit_log_dir), backend="file")


def _read_events(audit_log_dir) -> list:
    events = []
    for f in sorted(audit_log_dir.glob("audit_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# request_id
# ---------------------------------------------------------------------------

class TestRequestId:
    def test_request_id_injected_from_thread_local(self, audit_log_dir):
        set_request_id("test-rid-123")
        try:
            _logger(audit_log_dir).log("query_received", "sess1", {})
        finally:
            clear_request_id()
        events = _read_events(audit_log_dir)
        assert events[0]["request_id"] == "test-rid-123"

    def test_request_id_null_when_not_set(self, audit_log_dir):
        clear_request_id()
        _logger(audit_log_dir).log("query_received", "sess1", {})
        events = _read_events(audit_log_dir)
        assert events[0]["request_id"] is None


# ---------------------------------------------------------------------------
# pipeline_stage
# ---------------------------------------------------------------------------

class TestPipelineStage:
    def test_pipeline_stage_written(self, audit_log_dir):
        _logger(audit_log_dir).log("query_received", "s1", {}, pipeline_stage="input")
        events = _read_events(audit_log_dir)
        assert events[0]["pipeline_stage"] == "input"

    def test_pipeline_stage_null_when_omitted(self, audit_log_dir):
        _logger(audit_log_dir).log("query_received", "s1", {})
        events = _read_events(audit_log_dir)
        assert events[0]["pipeline_stage"] is None


# ---------------------------------------------------------------------------
# pii_masked event schema
# ---------------------------------------------------------------------------

class TestPIIMaskedSchema:
    def test_pii_masked_event_has_required_fields(self, audit_log_dir):
        _logger(audit_log_dir).log(
            "pii_masked",
            "sess1",
            {
                "types_found": ["US_SSN", "EMAIL_ADDRESS"],
                "count": 2,
                "recognizer_ids": ["presidio/spacy/en_core_web_lg", "presidio/pattern/email"],
            },
            pipeline_stage="input_masking",
        )
        events = _read_events(audit_log_dir)
        e = events[0]
        assert e["event_type"] == "pii_masked"
        assert e["pipeline_stage"] == "input_masking"
        assert "types_found" in e["details"]
        assert "count" in e["details"]
        assert "recognizer_ids" in e["details"]
        assert e["details"]["recognizer_ids"] != []


# ---------------------------------------------------------------------------
# output_pii_detected event schema
# ---------------------------------------------------------------------------

class TestOutputPIISchema:
    def test_output_pii_detected_has_required_fields(self, audit_log_dir):
        _logger(audit_log_dir).log(
            "output_pii_detected",
            "sess1",
            {
                "findings_count": 1,
                "info_types": ["US_SOCIAL_SECURITY_NUMBER"],
                "likelihoods": ["VERY_LIKELY"],
                "scan_duration_ms": 187.4,
            },
            pipeline_stage="output_scanning",
        )
        events = _read_events(audit_log_dir)
        e = events[0]
        assert e["event_type"] == "output_pii_detected"
        assert e["pipeline_stage"] == "output_scanning"
        d = e["details"]
        assert d["findings_count"] == 1
        assert "US_SOCIAL_SECURITY_NUMBER" in d["info_types"]
        assert "scan_duration_ms" in d


# ---------------------------------------------------------------------------
# Top-level envelope fields
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_all_required_fields_present(self, audit_log_dir):
        _logger(audit_log_dir).log("retrieval_done", "s1", {"result_count": 5}, pipeline_stage="retrieval")
        events = _read_events(audit_log_dir)
        e = events[0]
        for field in ("event_id", "timestamp", "event_type", "session_id", "request_id", "pipeline_stage", "details"):
            assert field in e, f"Missing field: {field}"

    def test_timestamp_is_utc_iso(self, audit_log_dir):
        _logger(audit_log_dir).log("generation_done", "s1", {})
        events = _read_events(audit_log_dir)
        ts = events[0]["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts


# ---------------------------------------------------------------------------
# Write failure does not raise
# ---------------------------------------------------------------------------

class TestStdoutBackend:
    def test_stdout_emits_valid_json(self, capsys):
        audit_logger = AuditLogger(backend="stdout")
        audit_logger.log("query_received", "s1", {"query_masked": "What is revenue?"})
        captured = capsys.readouterr()
        entry = json.loads(captured.out.strip())
        assert entry["event_type"] == "query_received"
        assert entry["session_id"] == "s1"

    def test_stdout_no_files_written(self, tmp_path, capsys):
        audit_logger = AuditLogger(log_dir=str(tmp_path), backend="stdout")
        audit_logger.log("query_received", "s1", {})
        assert list(tmp_path.glob("*.jsonl")) == []


class TestWriteFailure:
    def test_oserror_does_not_propagate(self, tmp_path):
        from unittest.mock import patch, mock_open
        audit_logger = AuditLogger(log_dir=str(tmp_path / "logs"))
        with patch("builtins.open", mock_open()) as m:
            m.side_effect = OSError("disk full")
            # must not raise
            event_id = audit_logger.log("query_received", "s1", {})
        assert event_id is not None
