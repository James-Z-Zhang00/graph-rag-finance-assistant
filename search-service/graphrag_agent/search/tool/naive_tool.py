"""
Industrialized vector similarity search tool.

Pipeline
--------
1. (Optional) Query rewriting — LLM expands the query for better recall.
   Disabled by default (NAIVE_QUERY_REWRITE_ENABLED=false) to keep latency low.
2. Hybrid retrieval — vector cosine search + BM25 full-text search run in
   parallel against the __Chunk__ node set in Neo4j.
3. RRF fusion — Reciprocal Rank Fusion (k=60) merges the two ranked lists
   into a single candidate pool without needing normalized scores.
4. Cross-encoder reranking — a local CrossEncoder model rescores the fused
   candidates with full query-document attention. Reduces the pool to top-K.

No graph traversal, no entity lookup, no community summaries.
"""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Neo4jVector
from langchain_core.tools import BaseTool

from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)
from graphrag_agent.config.settings import (
    NAIVE_QUERY_REWRITE_ENABLED,
    NAIVE_RERANK_ENABLED,
    NAIVE_RERANKER_MODEL,
    NAIVE_SEARCH_CANDIDATES,
    NAIVE_SEARCH_TOP_K,
    naive_description,
)
from graphrag_agent.search.tool.base import BaseSearchTool

# Lucene special characters that must be escaped in full-text queries
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

_QUERY_REWRITE_PROMPT = (
    "You are a financial document retrieval assistant. "
    "Rewrite the following question into a concise search query optimised for "
    "retrieving relevant passages from SEC filings and financial reports. "
    "Expand abbreviations, include synonyms, and keep it under 30 words. "
    "Return ONLY the rewritten query.\n\nQuestion: {query}\nRewritten query:"
)

_RRF_K = 60  # standard RRF constant — lower values favour top-ranked results


class NaiveSearchTool(BaseSearchTool):
    """
    Industrialized chunk retrieval:
      vector search + BM25 → RRF fusion → cross-encoder reranking.
    """

    def __init__(self):
        super().__init__(cache_dir="./cache/naive_search")
        self._last_results: List[RetrievalResult] = []
        self._vector_store: Optional[Neo4jVector] = None
        self._reranker = None
        self._setup_chains()

    # ── Setup ───────────────────────────────────────────────────────────────

    def _setup_chains(self) -> None:
        # 1. Vector store
        try:
            self._vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label="__Chunk__",
                text_node_properties=["text"],
                embedding_node_property="embedding",
            )
        except Exception as e:
            print(f"[NaiveSearchTool] Could not connect to chunk vector store: {e}")

        # 2. Full-text index (idempotent — IF NOT EXISTS)
        self._ensure_fulltext_index()

        # 3. Cross-encoder reranker
        if NAIVE_RERANK_ENABLED:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(NAIVE_RERANKER_MODEL)
                print(f"[NaiveSearchTool] Reranker loaded: {NAIVE_RERANKER_MODEL}")
            except Exception as e:
                print(f"[NaiveSearchTool] Could not load reranker ({e}); reranking disabled.")

    def _ensure_fulltext_index(self) -> None:
        try:
            with self.driver.session() as session:
                session.run(
                    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
                    "FOR (n:__Chunk__) ON EACH [n.text]"
                )
        except Exception as e:
            print(f"[NaiveSearchTool] Could not create full-text index: {e}")

    # ── Query rewriting (optional) ───────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            chain = PromptTemplate.from_template(_QUERY_REWRITE_PROMPT) | self.llm | StrOutputParser()
            rewritten = chain.invoke({"query": query}).strip()
            if rewritten:
                print(f"[NaiveSearchTool] Query rewritten: {rewritten!r}")
                return rewritten
        except Exception as e:
            print(f"[NaiveSearchTool] Query rewrite failed ({e}); using original.")
        return query

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, k: int) -> List[Tuple[str, str]]:
        """Returns [(chunk_id, text), ...] ordered by cosine similarity."""
        if self._vector_store is None:
            return []
        try:
            docs = self._vector_store.similarity_search(query, k=k)
            return [
                (doc.metadata.get("id", str(i)), doc.page_content)
                for i, doc in enumerate(docs)
            ]
        except Exception as e:
            print(f"[NaiveSearchTool] Vector search error: {e}")
            return []

    def _bm25_search(self, query: str, k: int) -> List[Tuple[str, str]]:
        """Returns [(chunk_id, text), ...] ordered by BM25 score."""
        # Escape Lucene special characters so the query doesn't throw a parse error
        safe_query = _LUCENE_SPECIAL.sub(r'\\\1', query)
        cypher = (
            "CALL db.index.fulltext.queryNodes('chunk_fulltext', $query) "
            "YIELD node, score "
            "RETURN coalesce(node.id, toString(id(node))) AS chunk_id, "
            "       node.text AS text, score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )
        try:
            results = self.db_query(cypher, {"query": safe_query, "k": k})
            if results is None or results.empty:
                return []
            return list(zip(results["chunk_id"].tolist(), results["text"].tolist()))
        except Exception as e:
            print(f"[NaiveSearchTool] BM25 search error: {e}")
            return []

    # ── RRF fusion ───────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        vector_hits: List[Tuple[str, str]],
        bm25_hits: List[Tuple[str, str]],
        k: int = _RRF_K,
    ) -> List[Tuple[str, str, float]]:
        """
        Reciprocal Rank Fusion.
        Returns [(chunk_id, text, rrf_score), ...] sorted descending.
        """
        scores: Dict[str, float] = defaultdict(float)
        texts: Dict[str, str] = {}

        for rank, (chunk_id, text) in enumerate(vector_hits):
            scores[chunk_id] += 1.0 / (k + rank + 1)
            texts[chunk_id] = text

        for rank, (chunk_id, text) in enumerate(bm25_hits):
            scores[chunk_id] += 1.0 / (k + rank + 1)
            texts.setdefault(chunk_id, text)  # vector text takes priority

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return [(cid, texts[cid], scores[cid]) for cid in ranked]

    # ── Reranking ────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        top_k: int,
    ) -> List[Tuple[str, str, float]]:
        """Cross-encoder reranking; falls back to RRF order on error."""
        if not self._reranker or not candidates:
            return candidates[:top_k]
        try:
            pairs = [(query, text) for _, text, _ in candidates]
            ce_scores = self._reranker.predict(pairs)
            ranked = sorted(
                zip(candidates, ce_scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            return [(cid, text, float(score)) for (cid, text, _), score in ranked[:top_k]]
        except Exception as e:
            print(f"[NaiveSearchTool] Reranking error ({e}); using RRF order.")
            return candidates[:top_k]

    # ── Public search interface ──────────────────────────────────────────────

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def search(self, query: str) -> str:
        self._last_results = []

        if self._vector_store is None:
            return "Chunk vector store is not available."

        # Step 1: optional query rewriting
        retrieval_query = self._rewrite_query(query) if NAIVE_QUERY_REWRITE_ENABLED else query

        # Step 2: parallel vector + BM25 retrieval
        vector_hits: List[Tuple[str, str]] = []
        bm25_hits: List[Tuple[str, str]] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vector = pool.submit(self._vector_search, retrieval_query, NAIVE_SEARCH_CANDIDATES)
            f_bm25 = pool.submit(self._bm25_search, retrieval_query, NAIVE_SEARCH_CANDIDATES)
            for future in as_completed([f_vector, f_bm25]):
                try:
                    result = future.result()
                    if future is f_vector:
                        vector_hits = result
                    else:
                        bm25_hits = result
                except Exception as e:
                    print(f"[NaiveSearchTool] Retrieval thread error: {e}")

        if not vector_hits and not bm25_hits:
            return "No relevant content found."

        # Step 3: RRF fusion
        fused = self._rrf_fuse(vector_hits, bm25_hits)

        # Step 4: cross-encoder reranking → top-K
        final = self._rerank(query, fused, NAIVE_SEARCH_TOP_K)

        # Build RetrievalResult objects for citation provenance
        self._last_results = [
            RetrievalResult(
                granularity="Chunk",
                evidence=text,
                metadata=RetrievalMetadata(
                    source_id=chunk_id,
                    source_type="chunk",
                    confidence=min(1.0, max(0.0, score)) if score <= 1.0 else 0.9,
                ),
                source="naive_search",
                score=score,
            )
            for chunk_id, text, score in final
        ]

        return "\n\n---\n\n".join(
            f"[Chunk {chunk_id}]\n{text}"
            for chunk_id, text, _ in final
        )

    def get_last_results(self) -> List[RetrievalResult]:
        return self._last_results

    def get_tool(self) -> BaseTool:
        search_fn = self.search

        class NaiveRAGTool(BaseTool):
            name: str = "naive_search"
            description: str = naive_description

            def _run(self_tool, query: Any) -> str:
                return search_fn(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("Async not supported")

        return NaiveRAGTool()
