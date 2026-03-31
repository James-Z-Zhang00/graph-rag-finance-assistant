"""
Cloud DLP output scanner — scans LLM-generated text for PII before it
reaches any caller (LLM06 mitigation).

Design decisions
----------------
* Uses inspect_content() (not deidentify_content()) — we detect, redact
  the spans ourselves, and log findings without sending raw PII to a
  deidentification endpoint.
* Synchronous scan() is safe to call from inside asyncio.to_thread().
* async_scan() wraps scan() via run_in_executor for use in async contexts
  (streaming path).
* Fail-open / fail-closed behaviour is controlled by DLP_FAIL_OPEN setting.
* Text larger than 100 KB is skipped with a warning — well below the DLP
  500 KB limit, chosen conservatively for chunked streaming content.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum text size sent to DLP (bytes). Responses exceeding this are skipped.
_MAX_TEXT_BYTES = 100_000


def _resolve_project_id(project_id: str) -> str:
    """
    Return project_id if explicitly set, otherwise attempt auto-detection
    from the GCP metadata server (available on Cloud Run, GCE, GKE).

    Falls back to empty string if not on GCP — DLPOutputScanner will then
    skip scanning gracefully rather than erroring.
    """
    if project_id:
        return project_id
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            detected = resp.read().decode().strip()
            logger.info("DLPOutputScanner: auto-detected GCP project_id=%s", detected)
            return detected
    except Exception:
        logger.debug("DLPOutputScanner: GCP metadata server not reachable — DLP disabled.")
        return ""

# Cloud DLP infoTypes targeting GLBA-relevant output risks
_DLP_INFO_TYPES = [
    {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "EMAIL_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "PERSON_NAME"},
    {"name": "STREET_ADDRESS"},
    {"name": "DATE_OF_BIRTH"},
    {"name": "US_BANK_ROUTING_MICR"},
    {"name": "IBAN_CODE"},
    {"name": "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"},
]


class DLPScanError(Exception):
    """Raised when DLP scan fails and DLP_FAIL_OPEN=False."""


@dataclass
class DLPFinding:
    info_type: str
    likelihood: str
    start_byte: int
    end_byte: int
    quote_masked: str = ""


@dataclass
class DLPScanResult:
    has_findings: bool
    findings: List[DLPFinding] = field(default_factory=list)
    redacted_text: str = ""
    scan_duration_ms: float = 0.0
    error: Optional[str] = None
    skipped: bool = False


class DLPOutputScanner:
    """
    Scans LLM output for PII using Google Cloud DLP before returning to caller.

    Parameters
    ----------
    project_id:      GCP project ID for DLP API calls.
    fail_open:       If True, return original text on any DLP error.
                     If False, raise DLPScanError.
    timeout_seconds: Maximum wall-clock seconds for a single DLP call.
    """

    def __init__(
        self,
        project_id: str,
        fail_open: bool = True,
        timeout_seconds: float = 3.0,
    ):
        self._project_id = _resolve_project_id(project_id)
        self._fail_open = fail_open
        self._timeout = timeout_seconds
        self._client = None
        self._parent = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dlp")

        if self._project_id:
            try:
                self._init_client()
            except Exception as exc:
                logger.warning("DLPOutputScanner: failed to init DLP client: %s", exc)

    def _init_client(self) -> None:
        import google.cloud.dlp_v2 as dlp_v2
        self._client = dlp_v2.DlpServiceClient()
        self._parent = f"projects/{self._project_id}"
        logger.info("DLPOutputScanner: Cloud DLP client initialised (project=%s).", self._project_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(
        self,
        text: str,
        session_id: str = "unknown",
        request_id: Optional[str] = None,
    ) -> DLPScanResult:
        """
        Synchronously scan *text* for PII.  Safe to call from a thread pool.

        Returns a DLPScanResult.  Never raises (respects fail_open policy).
        """
        if not self._project_id or self._client is None:
            return DLPScanResult(has_findings=False, redacted_text=text, skipped=True)

        if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
            logger.warning(
                "DLPOutputScanner: text exceeds %d bytes — skipping scan "
                "(session=%s, request_id=%s).",
                _MAX_TEXT_BYTES, session_id, request_id,
            )
            return DLPScanResult(has_findings=False, redacted_text=text, skipped=True)

        start = time.monotonic()
        try:
            future = self._executor.submit(self._call_dlp, text)
            response = future.result(timeout=self._timeout)
            duration_ms = (time.monotonic() - start) * 1000
            return self._build_result(text, response, duration_ms, session_id, request_id)

        except FuturesTimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "DLPOutputScanner: timeout after %.1fs (session=%s, request_id=%s).",
                self._timeout, session_id, request_id,
            )
            self._emit_audit("dlp_timeout", session_id, request_id, {
                "timeout_seconds": self._timeout,
                "scan_duration_ms": round(duration_ms, 2),
                "pipeline_stage": "output_scanning",
            })
            return self._on_error(text, f"DLP timeout after {self._timeout}s")

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "DLPOutputScanner: scan error (session=%s, request_id=%s): %s",
                session_id, request_id, exc,
            )
            self._emit_audit("dlp_scan_error", session_id, request_id, {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "scan_duration_ms": round(duration_ms, 2),
                "pipeline_stage": "output_scanning",
            })
            return self._on_error(text, str(exc))

    async def async_scan(
        self,
        text: str,
        session_id: str = "unknown",
        request_id: Optional[str] = None,
    ) -> DLPScanResult:
        """Async wrapper around scan() for use in the streaming path."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.scan(text, session_id=session_id, request_id=request_id),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_dlp(self, text: str) -> Any:
        """Execute the Cloud DLP inspect_content() gRPC call."""
        inspect_config = {
            "info_types": _DLP_INFO_TYPES,
            "min_likelihood": "POSSIBLE",
            "include_quote": False,
        }
        item = {"value": text}
        return self._client.inspect_content(
            request={"parent": self._parent, "inspect_config": inspect_config, "item": item}
        )

    def _build_result(
        self,
        original_text: str,
        response: Any,
        duration_ms: float,
        session_id: str,
        request_id: Optional[str],
    ) -> DLPScanResult:
        findings: List[DLPFinding] = []

        if response.result and response.result.findings:
            for f in response.result.findings:
                info_type = f.info_type.name if f.info_type else "UNKNOWN"
                likelihood = f.likelihood.name if f.likelihood else "UNKNOWN"
                byte_range = f.location.byte_range if f.location else None
                start_byte = byte_range.start if byte_range else 0
                end_byte = byte_range.end if byte_range else 0
                findings.append(DLPFinding(
                    info_type=info_type,
                    likelihood=likelihood,
                    start_byte=start_byte,
                    end_byte=end_byte,
                ))

        if not findings:
            return DLPScanResult(
                has_findings=False,
                findings=[],
                redacted_text=original_text,
                scan_duration_ms=round(duration_ms, 2),
            )

        redacted = self._redact_findings(original_text, findings)

        self._emit_audit("output_pii_detected", session_id, request_id, {
            "findings_count": len(findings),
            "info_types": [f.info_type for f in findings],
            "likelihoods": [f.likelihood for f in findings],
            "scan_duration_ms": round(duration_ms, 2),
            "pipeline_stage": "output_scanning",
        })

        return DLPScanResult(
            has_findings=True,
            findings=findings,
            redacted_text=redacted,
            scan_duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _redact_findings(text: str, findings: List[DLPFinding]) -> str:
        """Replace detected byte ranges with [DLP:REDACTED:{info_type}] placeholders."""
        encoded = text.encode("utf-8")
        # Sort descending so we replace from the end — avoids offset shifting
        sorted_findings = sorted(findings, key=lambda f: f.start_byte, reverse=True)
        for f in sorted_findings:
            placeholder = f"[DLP:REDACTED:{f.info_type}]".encode("utf-8")
            encoded = encoded[: f.start_byte] + placeholder + encoded[f.end_byte :]
        return encoded.decode("utf-8", errors="replace")

    def _on_error(self, text: str, error_msg: str) -> DLPScanResult:
        if not self._fail_open:
            raise DLPScanError(error_msg)
        return DLPScanResult(
            has_findings=False,
            redacted_text=text,
            error=error_msg,
        )

    @staticmethod
    def _emit_audit(
        event_type: str,
        session_id: str,
        request_id: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        """Best-effort audit log write — import here to avoid circular imports."""
        try:
            from graphrag_agent.compliance.audit_logger import AuditLogger
            from graphrag_agent.compliance.request_context import get_request_id
            AuditLogger().log(
                event_type,
                session_id,
                details,
                pipeline_stage=details.get("pipeline_stage"),
            )
        except Exception as exc:
            logger.error("DLPOutputScanner: failed to write audit event: %s", exc)
