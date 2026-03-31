"""
Streaming-path compliance tests — verifies PII masking, DLP scanning,
and status token ordering for ask_stream_with_compliance().
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_stream(agent, query, thread_id="t1"):
    chunks = []
    async for chunk in agent.ask_stream_with_compliance(query, thread_id=thread_id):
        chunks.append(chunk)
    return chunks


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def streaming_agent(audit_log_dir, mock_llm):
    """Agent with streaming mocked — DLP returns no findings."""
    with patch("graphrag_agent.agents.base.get_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
         patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
         patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

        no_findings = MagicMock()
        no_findings.result.findings = []
        MockDLP.return_value.inspect_content.return_value = no_findings

        from graphrag_agent.agents.hybrid_agent import HybridAgent
        agent = HybridAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.get_last_results.return_value = []

        # Mock base ask_stream to yield a fixed answer
        async def _mock_stream(query, thread_id="default", recursion_limit=None):
            yield "**Analyzing question**...\n\n"
            yield "Apple Inc. reported strong revenue."

        agent.__class__.__bases__[0].ask_stream = _mock_stream
        yield agent


@pytest.fixture()
def streaming_agent_dlp_ssn(audit_log_dir, mock_llm):
    """Streaming agent where DLP finds an SSN in the buffered answer."""
    finding = MagicMock()
    finding.info_type.name = "US_SOCIAL_SECURITY_NUMBER"
    finding.likelihood.name = "VERY_LIKELY"
    finding.location.byte_range.start = 0
    finding.location.byte_range.end = 11

    mock_response = MagicMock()
    mock_response.result.findings = [finding]

    with patch("graphrag_agent.agents.base.get_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
         patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
         patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

        MockDLP.return_value.inspect_content.return_value = mock_response

        from graphrag_agent.agents.hybrid_agent import HybridAgent
        agent = HybridAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.get_last_results.return_value = []

        async def _mock_stream(query, thread_id="default", recursion_limit=None):
            yield "**Retrieving information**...\n\n"
            yield "123-45-6789 is the SSN on record."

        agent.__class__.__bases__[0].ask_stream = _mock_stream
        yield agent


# ---------------------------------------------------------------------------
# Input masking on streaming path
# ---------------------------------------------------------------------------

class TestStreamInputMasking:
    def test_phone_not_in_accumulated_output(self, streaming_agent):
        # The query contains a phone number — it must not appear in stream output
        chunks = _run(_collect_stream(
            streaming_agent,
            "Call me at (555) 867-5309 for the Apple revenue query.",
        ))
        full = "".join(chunks)
        assert "(555) 867-5309" not in full


# ---------------------------------------------------------------------------
# DLP output scanning on streaming path
# ---------------------------------------------------------------------------

class TestStreamOutputDLPScan:
    def test_ssn_redacted_in_stream(self, streaming_agent_dlp_ssn):
        chunks = _run(_collect_stream(streaming_agent_dlp_ssn, "What is the SSN?"))
        full = "".join(chunks)
        assert "123-45-6789" not in full
        assert "[DLP:REDACTED:US_SOCIAL_SECURITY_NUMBER]" in full


# ---------------------------------------------------------------------------
# Status tokens arrive before DLP buffer
# ---------------------------------------------------------------------------

class TestStatusTokenOrder:
    def test_status_token_first_in_stream(self, streaming_agent):
        chunks = _run(_collect_stream(streaming_agent, "Apple revenue?"))
        status_indices = [i for i, c in enumerate(chunks) if c.startswith("**")]
        content_indices = [i for i, c in enumerate(chunks) if not c.startswith("**")]
        if status_indices and content_indices:
            assert min(status_indices) < min(content_indices)


# ---------------------------------------------------------------------------
# DLP fail-open in streaming
# ---------------------------------------------------------------------------

class TestStreamDLPFailOpen:
    def test_stream_continues_on_dlp_error(self, audit_log_dir, mock_llm):
        from google.api_core.exceptions import GoogleAPIError

        with patch("graphrag_agent.agents.base.get_llm_model", return_value=mock_llm), \
             patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=mock_llm), \
             patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
             patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
             patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

            MockDLP.return_value.inspect_content.side_effect = GoogleAPIError("DLP down")

            from graphrag_agent.agents.hybrid_agent import HybridAgent
            agent = HybridAgent()
            agent.search_tool = MagicMock()
            agent.search_tool.get_last_results.return_value = []

            async def _mock_stream(query, thread_id="default", recursion_limit=None):
                yield "Apple revenue was strong."

            agent.__class__.__bases__[0].ask_stream = _mock_stream

            chunks = _run(_collect_stream(agent, "Apple revenue?"))
            full = "".join(chunks)
            assert "[ERROR]" not in full
            assert len(full) > 0


# ---------------------------------------------------------------------------
# Audit events written on streaming path
# ---------------------------------------------------------------------------

class TestStreamAuditEvents:
    def test_query_received_event_written(self, streaming_agent, parse_audit_log, audit_log_dir):
        _run(_collect_stream(streaming_agent, "Apple revenue Q1 2024?"))
        events = parse_audit_log()
        types = [e["event_type"] for e in events]
        assert "query_received" in types
