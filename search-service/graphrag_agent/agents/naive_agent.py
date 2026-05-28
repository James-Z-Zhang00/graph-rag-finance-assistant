"""
NaiveAgent — plain vector similarity search with the same compliance layer as HybridAgent.

Skips graph traversal, entity lookup, and community summaries.
Embeds the query, finds top-K similar __Chunk__ nodes, and answers directly.
"""

import asyncio
import re
from typing import List, Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graphrag_agent.agents.base import BaseAgent
from graphrag_agent.agents.hybrid_agent import HybridAgent
from graphrag_agent.compliance import (
    PIIMasker,
    PresidioMasker,
    DLPOutputScanner,
    AuditLogger,
    HallucinationValidator,
)
from graphrag_agent.config.prompts import NAIVE_PROMPT, NAIVE_RAG_HUMAN_PROMPT
from graphrag_agent.config.settings import CACHE_DISABLED, COMPLIANCE_SETTINGS, response_type
from graphrag_agent.search.tool.naive_tool import NaiveSearchTool


class NaiveAgent(HybridAgent):
    """
    Naive RAG agent: embed → vector search → generate.

    Inherits the full compliance layer (PII masking, DLP, audit logging,
    hallucination validation) and LangGraph pipeline from HybridAgent.
    Overrides only the search tool and generate prompts.
    """

    def __init__(self):
        self.search_tool = NaiveSearchTool()
        self.cache_dir = "./cache/naive_agent"

        # Bootstrap base agent (sets up llm, embeddings, cache, tools, graph)
        # without going through HybridAgent.__init__ which would create a
        # HybridSearchTool we don't need.
        BaseAgent.__init__(self, cache_dir=self.cache_dir)

        # Compliance — identical setup to HybridAgent
        if COMPLIANCE_SETTINGS["presidio_enabled"]:
            self._pii_masker = PresidioMasker(
                score_threshold=COMPLIANCE_SETTINGS["presidio_score_threshold"]
            )
        else:
            self._pii_masker = PIIMasker()

        self.audit_logger = AuditLogger()
        self._validator = HallucinationValidator(llm=self.llm)

        self._dlp_scanner: Optional[DLPOutputScanner] = None
        if COMPLIANCE_SETTINGS["dlp_enabled"]:
            self._dlp_scanner = DLPOutputScanner(
                project_id=COMPLIANCE_SETTINGS["dlp_project_id"],
                fail_open=COMPLIANCE_SETTINGS["dlp_fail_open"],
                timeout_seconds=COMPLIANCE_SETTINGS["dlp_timeout_seconds"],
            )

        self._last_citations: List[str] = []
        self._last_quality_score: Optional[float] = None
        self._last_audit_id: Optional[str] = None
        self._last_contexts: List[str] = []

    # ── Tool + keyword setup ────────────────────────────────────────────────

    def _setup_tools(self) -> List:
        return [self.search_tool.get_tool()]

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    # ── Generate node (non-streaming) ───────────────────────────────────────

    def _generate_node(self, state):
        messages = state["messages"]
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        self._last_citations = self._build_citations()
        self._last_contexts = [docs] if docs and docs != "No relevant information found" else []

        last_results = self.search_tool.get_last_results()
        self.audit_logger.log("retrieval_done", thread_id, {
            "result_count": len(last_results),
            "local_count": len(last_results),
            "global_count": 0,
        }, pipeline_stage="retrieval")

        global_result = None if CACHE_DISABLED else self.global_cache_manager.get(question)
        if global_result and isinstance(global_result, str):
            return {"messages": [AIMessage(content=global_result)]}

        cached_result = None if CACHE_DISABLED else self.cache_manager.get(question, thread_id=thread_id)
        if cached_result and isinstance(cached_result, str):
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", NAIVE_RAG_HUMAN_PROMPT),
        ])
        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs,
                "question": question,
                "response_type": response_type,
            })

            if not CACHE_DISABLED and response and len(response) > 10:
                self.cache_manager.set(question, response, thread_id=thread_id)
                self.global_cache_manager.set(question, response)

            self.audit_logger.log("generation_done", thread_id, {
                "answer_length": len(response) if response else 0,
            }, pipeline_stage="generation")

            self._log_execution("generate", {"question": question, "docs_length": len(docs)}, response)
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            self._log_execution("generate_error", {"question": question}, str(e))
            return {"messages": [AIMessage(content=f"Sorry, I am unable to answer this question. Technical reason: {str(e)}")]}

    # ── Generate node (streaming) ───────────────────────────────────────────

    async def _generate_node_stream(self, state):
        messages = state["messages"]
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result and not isinstance(cached_result, str):
            cached_result = None
        if cached_result:
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            for i, piece in enumerate(chunks):
                buffer += piece
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            if buffer:
                yield buffer
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", NAIVE_RAG_HUMAN_PROMPT),
        ])
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,
            "question": question,
            "response_type": response_type,
        })

        if response is None:
            response = "Unable to generate a response. Please try rephrasing your question."

        sentences = re.split(r'([.!?。！？]\s*)', response)
        buffer = ""
        for i, piece in enumerate(sentences):
            buffer += piece
            if i % 2 == 1 or len(buffer) >= self.stream_flush_threshold:
                yield buffer
                buffer = ""
                await asyncio.sleep(0.01)
        if buffer:
            yield buffer

    def close(self):
        BaseAgent.close(self)
        if self.search_tool:
            self.search_tool.close()
