"""
Router-level compliance tests using FastAPI TestClient.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture()
def client(audit_log_dir):
    """FastAPI TestClient with mocked agent pool."""
    with patch("graphrag_agent.agents.base.get_llm_model", return_value=MagicMock()), \
         patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=MagicMock()), \
         patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
         patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
         patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

        no_findings = MagicMock()
        no_findings.result.findings = []
        MockDLP.return_value.inspect_content.return_value = no_findings

        mock_agent = MagicMock()
        mock_agent.ask_with_trace.return_value = {
            "answer": "Apple revenue was $119B.",
            "execution_log": [],
            "citations": [],
            "quality_score": 0.9,
            "audit_id": "test-audit-id",
            "contexts": [],
        }

        async def _mock_compliance_stream(query, thread_id="default", recursion_limit=None):
            yield "Apple revenue was $119B."

        mock_agent.ask_stream_with_compliance = _mock_compliance_stream

        with patch("hybrid_search_agent.agent.agent_pool") as mock_pool:
            mock_pool.get.return_value = mock_agent

            from fastapi.testclient import TestClient
            from main import app
            yield TestClient(app)


class TestHybridEndpoint:
    def test_post_returns_200(self, client):
        resp = client.post("/search/hybrid", json={"query": "Apple revenue?", "session_id": "s1"})
        assert resp.status_code == 200

    def test_answer_field_present(self, client):
        resp = client.post("/search/hybrid", json={"query": "Apple revenue?", "session_id": "s1"})
        assert "answer" in resp.json()

    def test_ask_with_trace_called_not_ask(self, audit_log_dir):
        """Verify the router always calls ask_with_trace (compliance path), not ask()."""
        with patch("graphrag_agent.agents.base.get_llm_model", return_value=MagicMock()), \
             patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=MagicMock()), \
             patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
             patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
             patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

            MockDLP.return_value.inspect_content.return_value = MagicMock(result=MagicMock(findings=[]))

            mock_agent = MagicMock()
            mock_agent.ask_with_trace.return_value = {
                "answer": "test", "execution_log": [], "citations": [],
                "quality_score": 1.0, "audit_id": "x", "contexts": [],
            }

            with patch("hybrid_search_agent.agent.agent_pool") as mock_pool:
                mock_pool.get.return_value = mock_agent
                from fastapi.testclient import TestClient
                from main import app
                c = TestClient(app)
                c.post("/search/hybrid", json={"query": "test", "session_id": "s1"})

            mock_agent.ask_with_trace.assert_called_once()
            mock_agent.ask.assert_not_called()


class TestStreamEndpoint:
    def test_stream_endpoint_returns_200(self, client):
        resp = client.post("/search/hybrid/stream", json={"query": "Apple revenue?", "session_id": "s1"})
        assert resp.status_code == 200

    def test_stream_calls_compliance_method_not_base(self, audit_log_dir):
        """Verify the stream router calls ask_stream_with_compliance, not ask_stream."""
        with patch("graphrag_agent.agents.base.get_llm_model", return_value=MagicMock()), \
             patch("graphrag_agent.agents.base.get_stream_llm_model", return_value=MagicMock()), \
             patch("graphrag_agent.agents.base.get_embeddings_model", return_value=MagicMock()), \
             patch("graphrag_agent.search.tool.hybrid_tool.HybridSearchTool", autospec=True), \
             patch("google.cloud.dlp_v2.DlpServiceClient") as MockDLP:

            MockDLP.return_value.inspect_content.return_value = MagicMock(result=MagicMock(findings=[]))

            mock_agent = MagicMock()

            async def _compliance_stream(query, thread_id="default", recursion_limit=None):
                yield "test"

            mock_agent.ask_stream_with_compliance = _compliance_stream
            mock_agent.ask_stream = AsyncMock()

            with patch("hybrid_search_agent.agent.agent_pool") as mock_pool:
                mock_pool.get.return_value = mock_agent
                from fastapi.testclient import TestClient
                from main import app
                c = TestClient(app)
                c.post("/search/hybrid/stream", json={"query": "test", "session_id": "s1"})

            mock_agent.ask_stream.assert_not_called()
