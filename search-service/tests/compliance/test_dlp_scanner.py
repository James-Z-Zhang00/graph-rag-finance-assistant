"""
Unit tests for DLPOutputScanner.
All tests mock the Cloud DLP client — no live API calls.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from graphrag_agent.compliance.dlp_scanner import DLPOutputScanner, DLPScanError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner(mock_client=None, fail_open=True, timeout=3.0):
    with patch("google.cloud.dlp_v2.DlpServiceClient") as MockClient:
        if mock_client is not None:
            MockClient.return_value = mock_client
        scanner = DLPOutputScanner(
            project_id="test-project",
            fail_open=fail_open,
            timeout_seconds=timeout,
        )
    return scanner


def _no_findings_client():
    mock_response = MagicMock()
    mock_response.result.findings = []
    client = MagicMock()
    client.inspect_content.return_value = mock_response
    return client


def _ssn_finding_client(start=0, end=11):
    finding = MagicMock()
    finding.info_type.name = "US_SOCIAL_SECURITY_NUMBER"
    finding.likelihood.name = "VERY_LIKELY"
    finding.location.byte_range.start = start
    finding.location.byte_range.end = end
    mock_response = MagicMock()
    mock_response.result.findings = [finding]
    client = MagicMock()
    client.inspect_content.return_value = mock_response
    return client


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_no_findings_returns_original(self):
        scanner = _make_scanner(_no_findings_client())
        result = scanner.scan("Apple revenue was $119B in Q1 2024.")
        assert result.has_findings is False
        assert result.redacted_text == "Apple revenue was $119B in Q1 2024."

    def test_no_findings_empty_list(self):
        scanner = _make_scanner(_no_findings_client())
        result = scanner.scan("Clean text here.")
        assert result.findings == []


# ---------------------------------------------------------------------------
# SSN finding
# ---------------------------------------------------------------------------

class TestSSNFinding:
    def test_ssn_detected_and_redacted(self):
        text = "123-45-6789 is the SSN."
        scanner = _make_scanner(_ssn_finding_client(start=0, end=11))
        result = scanner.scan(text)
        assert result.has_findings is True
        assert "123-45-6789" not in result.redacted_text
        assert "[DLP:REDACTED:US_SOCIAL_SECURITY_NUMBER]" in result.redacted_text

    def test_ssn_finding_in_findings_list(self):
        scanner = _make_scanner(_ssn_finding_client())
        result = scanner.scan("123-45-6789")
        assert len(result.findings) == 1
        assert result.findings[0].info_type == "US_SOCIAL_SECURITY_NUMBER"
        assert result.findings[0].likelihood == "VERY_LIKELY"


# ---------------------------------------------------------------------------
# Multiple findings
# ---------------------------------------------------------------------------

class TestMultipleFindings:
    def test_two_findings_both_redacted(self):
        text = "123-45-6789 and 987-65-4321 are SSNs."
        # two findings at known byte offsets
        f1 = MagicMock()
        f1.info_type.name = "US_SOCIAL_SECURITY_NUMBER"
        f1.likelihood.name = "VERY_LIKELY"
        f1.location.byte_range.start = 0
        f1.location.byte_range.end = 11

        f2 = MagicMock()
        f2.info_type.name = "US_SOCIAL_SECURITY_NUMBER"
        f2.likelihood.name = "VERY_LIKELY"
        f2.location.byte_range.start = 16
        f2.location.byte_range.end = 27

        mock_response = MagicMock()
        mock_response.result.findings = [f1, f2]
        client = MagicMock()
        client.inspect_content.return_value = mock_response

        scanner = _make_scanner(client)
        result = scanner.scan(text)
        assert result.has_findings is True
        assert len(result.findings) == 2
        assert "123-45-6789" not in result.redacted_text
        assert "987-65-4321" not in result.redacted_text


# ---------------------------------------------------------------------------
# Audit log written on finding
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_event_written_on_finding(self, audit_log_dir, parse_audit_log):
        scanner = _make_scanner(_ssn_finding_client())
        scanner.scan("123-45-6789", session_id="sess1")
        events = parse_audit_log()
        types = [e["event_type"] for e in events]
        assert "output_pii_detected" in types

    def test_audit_event_fields(self, audit_log_dir, parse_audit_log):
        scanner = _make_scanner(_ssn_finding_client())
        scanner.scan("123-45-6789", session_id="sess1", request_id="req-abc")
        events = [e for e in parse_audit_log() if e["event_type"] == "output_pii_detected"]
        assert len(events) == 1
        details = events[0]["details"]
        assert details["findings_count"] == 1
        assert "US_SOCIAL_SECURITY_NUMBER" in details["info_types"]
        assert "scan_duration_ms" in details


# ---------------------------------------------------------------------------
# Error handling — fail-open
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_api_error_fail_open_returns_original(self):
        from google.api_core.exceptions import GoogleAPIError
        client = MagicMock()
        client.inspect_content.side_effect = GoogleAPIError("DLP down")
        scanner = _make_scanner(client, fail_open=True)
        result = scanner.scan("Some text with 123-45-6789.")
        assert result.error is not None
        assert result.redacted_text == "Some text with 123-45-6789."

    def test_api_error_fail_open_does_not_raise(self):
        from google.api_core.exceptions import GoogleAPIError
        client = MagicMock()
        client.inspect_content.side_effect = GoogleAPIError("DLP down")
        scanner = _make_scanner(client, fail_open=True)
        scanner.scan("text")  # must not raise


# ---------------------------------------------------------------------------
# Error handling — fail-closed
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_api_error_fail_closed_raises(self):
        from google.api_core.exceptions import GoogleAPIError
        client = MagicMock()
        client.inspect_content.side_effect = GoogleAPIError("DLP down")
        scanner = _make_scanner(client, fail_open=False)
        with pytest.raises(DLPScanError):
            scanner.scan("text with 123-45-6789")


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_fail_open_returns_original(self):
        import time
        client = MagicMock()
        client.inspect_content.side_effect = lambda **kw: time.sleep(10)
        scanner = _make_scanner(client, fail_open=True, timeout=0.1)
        result = scanner.scan("text")
        assert result.redacted_text == "text"
        assert result.error is not None

    def test_timeout_audit_event_written(self, audit_log_dir, parse_audit_log):
        import time
        client = MagicMock()
        client.inspect_content.side_effect = lambda **kw: time.sleep(10)
        scanner = _make_scanner(client, fail_open=True, timeout=0.1)
        scanner.scan("text", session_id="s1")
        events = parse_audit_log()
        types = [e["event_type"] for e in events]
        assert "dlp_timeout" in types


# ---------------------------------------------------------------------------
# DLP disabled
# ---------------------------------------------------------------------------

class TestDLPDisabled:
    def test_no_project_id_is_noop(self):
        scanner = DLPOutputScanner(project_id="", fail_open=True)
        result = scanner.scan("123-45-6789")
        assert result.skipped is True
        assert result.redacted_text == "123-45-6789"


# ---------------------------------------------------------------------------
# Oversized text
# ---------------------------------------------------------------------------

class TestOversizedText:
    def test_oversized_text_skipped(self):
        scanner = _make_scanner(_no_findings_client())
        huge_text = "a" * 200_000  # 200 KB
        result = scanner.scan(huge_text)
        assert result.skipped is True
        assert result.redacted_text == huge_text


# ---------------------------------------------------------------------------
# Async scan
# ---------------------------------------------------------------------------

class TestAsyncScan:
    def test_async_scan_returns_same_result(self):
        scanner = _make_scanner(_ssn_finding_client())

        async def _run():
            return await scanner.async_scan("123-45-6789", session_id="s1")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.has_findings is True
        assert "[DLP:REDACTED:US_SOCIAL_SECURITY_NUMBER]" in result.redacted_text
