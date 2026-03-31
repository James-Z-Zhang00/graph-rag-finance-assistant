"""
Shared pytest fixtures for the compliance test suite.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Environment / settings overrides
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_compliance_settings(monkeypatch):
    """Ensure tests never hit real Presidio model load or real DLP API."""
    monkeypatch.setenv("DLP_PROJECT_ID", "test-project")
    monkeypatch.setenv("AUDIT_LOG_DIR", "/tmp/test_audit_logs")


@pytest.fixture()
def compliance_env(monkeypatch):
    """Return a helper that sets compliance env vars per-test."""
    def _set(**kwargs):
        mapping = {
            "presidio_enabled": "PRESIDIO_ENABLED",
            "dlp_enabled": "DLP_ENABLED",
            "dlp_fail_open": "DLP_FAIL_OPEN",
            "dlp_project_id": "DLP_PROJECT_ID",
            "dlp_timeout_seconds": "DLP_TIMEOUT_SECONDS",
            "presidio_score_threshold": "PRESIDIO_SCORE_THRESHOLD",
        }
        for key, value in kwargs.items():
            env_key = mapping.get(key, key.upper())
            monkeypatch.setenv(env_key, str(value).lower() if isinstance(value, bool) else str(value))
    return _set


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def audit_log_dir(tmp_path) -> Path:
    """Temporary directory for audit logs; patches AuditLogger to use it."""
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir()
    with patch.dict(os.environ, {"AUDIT_LOG_DIR": str(log_dir)}):
        yield log_dir


@pytest.fixture()
def parse_audit_log(audit_log_dir):
    """Return a callable that reads all JSONL events from the temp audit dir."""
    def _parse():
        events = []
        for f in sorted(audit_log_dir.glob("audit_*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return events
    return _parse


# ---------------------------------------------------------------------------
# Mock DLP client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_dlp_no_findings():
    """DLP client that always returns no findings."""
    mock_response = MagicMock()
    mock_response.result.findings = []
    mock_client = MagicMock()
    mock_client.inspect_content.return_value = mock_response
    with patch("google.cloud.dlp_v2.DlpServiceClient", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def mock_dlp_with_ssn():
    """DLP client that returns a single US_SOCIAL_SECURITY_NUMBER finding."""
    finding = MagicMock()
    finding.info_type.name = "US_SOCIAL_SECURITY_NUMBER"
    finding.likelihood.name = "VERY_LIKELY"
    finding.location.byte_range.start = 0
    finding.location.byte_range.end = 11  # len("123-45-6789")

    mock_response = MagicMock()
    mock_response.result.findings = [finding]
    mock_client = MagicMock()
    mock_client.inspect_content.return_value = mock_response
    with patch("google.cloud.dlp_v2.DlpServiceClient", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def mock_dlp_error():
    """DLP client that raises GoogleAPIError on inspect_content."""
    from google.api_core.exceptions import GoogleAPIError
    mock_client = MagicMock()
    mock_client.inspect_content.side_effect = GoogleAPIError("simulated DLP error")
    with patch("google.cloud.dlp_v2.DlpServiceClient", return_value=mock_client):
        yield mock_client


# ---------------------------------------------------------------------------
# Presidio masker (module-scoped — avoids 2-4s NLP model load per test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def presidio_masker_instance():
    """
    Load PresidioMasker once per test module.

    If spaCy / Presidio are not installed, falls back to regex mode
    so unit tests still run in lightweight CI environments.
    """
    from graphrag_agent.compliance.presidio_masker import PresidioMasker
    return PresidioMasker(score_threshold=0.5)


# ---------------------------------------------------------------------------
# Sample PII texts keyed by GLBA entity type
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_pii_texts():
    return {
        "EMAIL_ADDRESS": "Contact me at john.doe@example.com for details.",
        "PHONE_NUMBER": "Call us at (555) 867-5309 or 555.867.5309.",
        "US_SSN": "My SSN is 123-45-6789.",
        "US_SSN_SPACE": "My SSN is 123 45 6789.",
        "CREDIT_CARD_NUMBER": "Card number: 4111 1111 1111 1111.",
        "PERSON_NAME": "John Smith submitted the report.",
        "STREET_ADDRESS": "She lives at 123 Main Street, Boston, MA 02101.",
        "DATE_OF_BIRTH": "Born on 01/15/1985.",
        "FINANCIAL_ACCOUNT_NUMBER": "routing number 021000021 for the transfer.",
        "CLEAN": "What was Apple Inc. revenue in Q1 2024?",
    }


# ---------------------------------------------------------------------------
# Mock LLM stub
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm():
    """Deterministic LLM stub returning a fixed answer."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Apple Inc. reported $119.6 billion in revenue for Q1 2024."
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm
