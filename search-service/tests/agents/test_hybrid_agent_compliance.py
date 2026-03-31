"""
Integration tests for the full compliance pipeline in HybridAgent.
Mocks LLM and Neo4j; uses real Presidio masker and mocked DLP.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_with_mocks(audit_log_dir, mock_llm):
    """HybridAgent with mocked LLM, search tool, and graph — real compliance layer."""
    with patch("graphrag_agent.agents.base.get_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=mock_llm), \
         patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
         patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
         patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

        # DLP returns no findings by default
        no_findings = MagicMock()
        no_findings.result.findings = []
        MockDLP.return_value.inspect_content.return_value = no_findings

        from graphrag_agent.agents.hybrid_agent import HybridAgent
        agent = HybridAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.get_last_results.return_value = []
        agent.search_tool.get_tool.return_value = MagicMock()
        agent.search_tool.get_global_tool.return_value = MagicMock()
        yield agent


@pytest.fixture()
def agent_dlp_with_ssn(audit_log_dir, mock_llm):
    """Agent where DLP returns an SSN finding in the LLM output."""
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
        agent.search_tool.get_tool.return_value = MagicMock()
        agent.search_tool.get_global_tool.return_value = MagicMock()

        # LLM returns answer starting with fake SSN
        resp = MagicMock()
        resp.content = "123-45-6789 was the reference number."
        mock_llm.invoke.return_value = resp
        yield agent


# ---------------------------------------------------------------------------
# Boundary 1: input masking
# ---------------------------------------------------------------------------

class TestInputMasking:
    def test_pii_stripped_from_query_before_llm(self, agent_with_mocks, audit_log_dir):
        """The raw email must never reach the LLM."""
        captured_messages = []

        original_invoke = agent_with_mocks.llm.invoke

        def capture(messages, **kw):
            captured_messages.extend(messages if isinstance(messages, list) else [messages])
            return original_invoke(messages, **kw)

        agent_with_mocks.llm.invoke = capture

        with patch.object(agent_with_mocks, "_setup_graph"):
            pass

        agent_with_mocks.ask_with_trace(
            "Revenue question from john.doe@example.com", thread_id="t1"
        )

        for msg in captured_messages:
            content = getattr(msg, "content", str(msg))
            assert "john.doe@example.com" not in content


# ---------------------------------------------------------------------------
# Boundary 2: DLP output scan
# ---------------------------------------------------------------------------

class TestOutputDLPScan:
    def test_ssn_redacted_in_answer(self, agent_dlp_with_ssn):
        result = agent_dlp_with_ssn.ask_with_trace("What is the ref number?", thread_id="t1")
        assert "123-45-6789" not in result["answer"]
        assert "[DLP:REDACTED:US_SOCIAL_SECURITY_NUMBER]" in result["answer"]


# ---------------------------------------------------------------------------
# request_id consistency
# ---------------------------------------------------------------------------

class TestRequestIdConsistency:
    def test_all_events_share_request_id(self, agent_with_mocks, audit_log_dir, parse_audit_log):
        agent_with_mocks.ask_with_trace("What was Apple revenue?", thread_id="t1")
        events = parse_audit_log()
        request_ids = {e["request_id"] for e in events if e["request_id"] is not None}
        assert len(request_ids) == 1, f"Expected 1 unique request_id, got: {request_ids}"


# ---------------------------------------------------------------------------
# Pipeline stage coverage
# ---------------------------------------------------------------------------

class TestPipelineStages:
    def test_expected_stages_logged(self, agent_with_mocks, parse_audit_log, audit_log_dir):
        agent_with_mocks.ask_with_trace("Apple revenue Q1 2024?", thread_id="t1")
        stages = {e["pipeline_stage"] for e in parse_audit_log() if e["pipeline_stage"]}
        assert "input" in stages


# ---------------------------------------------------------------------------
# DLP failure — fail-open
# ---------------------------------------------------------------------------

class TestDLPFailureFailOpen:
    def test_dlp_error_does_not_fail_request(self, audit_log_dir, mock_llm):
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

            result = agent.ask_with_trace("Apple revenue?", thread_id="t1")
            assert "answer" in result


# ---------------------------------------------------------------------------
# Clean query — no pii_masked event
# ---------------------------------------------------------------------------

class TestCleanQuery:
    def test_no_pii_masked_event_for_clean_query(self, agent_with_mocks, parse_audit_log, audit_log_dir):
        agent_with_mocks.ask_with_trace("What was Apple revenue in Q1 2024?", thread_id="t1")
        events = parse_audit_log()
        pii_events = [e for e in events if e["event_type"] == "pii_masked"]
        assert pii_events == []
