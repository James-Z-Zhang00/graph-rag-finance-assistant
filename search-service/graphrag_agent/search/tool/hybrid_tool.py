import re
import time
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from neo4j import Result

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    HYBRID_TOOL_QUERY_PROMPT,
    LOCAL_SEARCH_KEYWORD_PROMPT,
)
from graphrag_agent.config.settings import gl_description, response_type, HYBRID_SEARCH_SETTINGS
from graphrag_agent.pipelines.sec.models import FINANCIAL_FACT_WRITE_FIELDS
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.search.retrieval_adapter import (
    create_retrieval_metadata,
    create_retrieval_result,
    merge_retrieval_results,
    results_from_entities,
    results_from_relationships,
    results_to_payload,
)


class HybridSearchTool(BaseSearchTool):
    """
    Hybrid search tool implementing a two-level retrieval strategy similar to LightRAG.
    Combines local detail retrieval with global thematic retrieval.
    """

    def __init__(self):
        """Initialize the hybrid search tool."""
        # Retrieval parameters
        self.entity_limit = HYBRID_SEARCH_SETTINGS["entity_limit"]
        self.max_hop_distance = HYBRID_SEARCH_SETTINGS["max_hop_distance"]
        self.top_communities = HYBRID_SEARCH_SETTINGS["top_communities"]
        self.batch_size = HYBRID_SEARCH_SETTINGS["batch_size"]
        self.community_level = HYBRID_SEARCH_SETTINGS["community_level"]

        # Call parent constructor
        super().__init__(cache_dir="./cache/hybrid_search")

        # Set up processing chains
        self._setup_chains()

    def _setup_chains(self):
        """Set up LLM processing chains."""
        # Main query chain - used to generate the final answer
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_TOOL_QUERY_PROMPT),
        ])

        # Connect to LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()

        # Keyword extraction chain
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", LOCAL_SEARCH_KEYWORD_PROMPT),
            ("human", "{query}"),
        ])

        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Extract two-level keywords from a query.

        Args:
            query: Query string

        Returns:
            Dict[str, List[str]]: Categorized keyword dictionary
        """
        # Check cache
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords

        try:
            llm_start = time.time()

            # Call LLM to extract keywords
            result = self.keyword_chain.invoke({"query": query})

            print(f"DEBUG - LLM keyword result: {result[:100]}...") if len(str(result)) > 100 else print(f"DEBUG - LLM keyword result: {result}")

            # Parse JSON result
            try:
                # Try direct parse
                if isinstance(result, dict):
                    # Result is already a dict, no parsing needed
                    keywords = result
                elif isinstance(result, str):
                    # Clean string, remove characters that may cause parse failure
                    result = result.strip()
                    # Check if string starts in JSON format
                    if result.startswith('{') and result.endswith('}'):
                        keywords = json.loads(result)
                    else:
                        # Try to extract JSON portion - find first { and last }
                        start_idx = result.find('{')
                        end_idx = result.rfind('}')
                        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                            json_str = result[start_idx:end_idx+1]
                            keywords = json.loads(json_str)
                        else:
                            # No valid JSON structure, use simple keyword extraction
                            raise ValueError("No valid JSON structure found")
                else:
                    # Neither string nor dict
                    raise TypeError(f"Unexpected result type: {type(result)}")

            except (json.JSONDecodeError, ValueError, TypeError) as json_err:
                print(f"JSON parse failed: {json_err}, attempting fallback keyword extraction")

                # Fallback: manual keyword extraction
                if isinstance(result, str):
                    # Simple tokenization for keyword extraction
                    import re
                    # Remove punctuation, split by whitespace
                    words = re.findall(r'\b\w+\b', query.lower())
                    # Filter stopwords (simplified; a fuller stopword list would be better)
                    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                                "in", "on", "at", "to", "for", "with", "by", "about", "of", "and", "or"}
                    keywords = {
                        "high_level": [word for word in words if len(word) > 5 and word not in stopwords][:3],
                        "low_level": [word for word in words if 3 <= len(word) <= 5 and word not in stopwords][:5]
                    }
                else:
                    # If not a string, return simple keywords based on original query
                    keywords = {
                        "high_level": [query],
                        "low_level": []
                    }

            # Record LLM processing time
            self.performance_metrics["llm_time"] += time.time() - llm_start

            # Ensure required keys are present
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []

            # Ensure list types
            if not isinstance(keywords["low_level"], list):
                keywords["low_level"] = [str(keywords["low_level"])]
            if not isinstance(keywords["high_level"], list):
                keywords["high_level"] = [str(keywords["high_level"])]

            # Cache result
            self.cache_manager.set(f"keywords:{query}", keywords)

            return keywords

        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            # Return default value based on original query
            return {"low_level": [query], "high_level": [query.split()[0] if query.split() else query]}

    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            pandas.DataFrame: Query results
        """
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )

    def _vector_search(self, query: str, limit: int = 5) -> List[str]:
        """
        Delegate to the base class vector search method.

        Args:
            query: Query string
            limit: Maximum number of results

        Returns:
            List[str]: Entity ID list
        """
        return self.vector_search(query, limit)

    def _fallback_text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        Fallback text-matching search method.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List[str]: Matching entity ID list
        """
        try:
            # Full-text search query
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """

            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })

            if not results.empty:
                return results['id'].tolist()
            else:
                return []

        except Exception as e:
            print(f"Text search also failed: {e}")
            return []

    def _retrieve_low_level_content(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve low-level content (specific entities and relationships).

        Args:
            query: Query string
            keywords: Low-level keyword list

        Returns:
            Tuple[str, List[RetrievalResult]]: Formatted content and corresponding evidence
        """
        query_start = time.time()
        retrieval_results: List[RetrievalResult] = []

        # First use keyword query to retrieve relevant entities
        entity_ids = []

        if keywords:
            keyword_params = {}
            keyword_conditions = []

            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_params[param_name] = keyword
                keyword_conditions.append(f"e.id CONTAINS ${param_name} OR e.description CONTAINS ${param_name}")

            # Build query
            if keyword_conditions:
                keyword_query = """
                MATCH (e:__Entity__)
                WHERE """ + " OR ".join(keyword_conditions) + """
                RETURN e.id AS id
                LIMIT $limit
                """

                try:
                    keyword_results = self.db_query(keyword_query,
                                                {**keyword_params, "limit": self.entity_limit})
                    if not keyword_results.empty:
                        entity_ids = keyword_results['id'].tolist()
                except Exception as e:
                    print(f"Keyword query failed: {e}")

        # If keyword search returns no results or no keywords provided, try vector search
        if not entity_ids:
            try:
                vector_entity_ids = self._vector_search(query, limit=self.entity_limit)
                if vector_entity_ids:
                    entity_ids = vector_entity_ids
            except Exception as e:
                print(f"Vector search failed: {e}")

        # If still no entities, use basic text matching
        if not entity_ids:
            try:
                entity_ids = self._fallback_text_search(query, limit=self.entity_limit)
            except Exception as e:
                print(f"Text search failed: {e}")

        # If still no entities, return empty content
        if not entity_ids:
            self.performance_metrics["query_time"] += time.time() - query_start
            return "No relevant low-level content found.", retrieval_results

        # Fetch entity info - avoid multi-hop relationships to keep queries simple
        entity_query = """
        // Start from seed entities
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids

        RETURN collect({
            id: e.id,
            type: CASE WHEN size(labels(e)) > 1
                     THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0]
                     ELSE 'Unknown'
                  END,
            description: e.description
        }) AS entities
        """

        # Fetch relationship info - separate query to avoid complex path queries
        relation_query = """
        // Find relationships between entities
        MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        WHERE e1.id IN $entity_ids
          AND e2.id IN $entity_ids
          AND e1.id < e2.id  // avoid duplicate relationships

        RETURN collect({
            start: e1.id,
            type: type(r),
            end: e2.id,
            description: CASE WHEN r.description IS NULL THEN '' ELSE r.description END
        }) AS relationships
        """

        # Fetch chunk info
        chunk_query = """
        // Find text chunks that mention these entities
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE e.id IN $entity_ids

        RETURN collect(DISTINCT {
            id: c.id,
            text: c.text
        })[0..5] AS chunks
        """

        # Fallback: text-match on chunk content using entity names (used when MENTIONS is missing)
        chunk_fallback_query = """
        MATCH (c:__Chunk__)
        WHERE ANY(eid IN $entity_ids WHERE toLower(c.text) CONTAINS toLower(eid))
        RETURN collect(DISTINCT {
            id: c.id,
            text: c.text
        })[0..5] AS chunks
        """

        try:
            # Fetch entity info
            entity_results = self.db_query(entity_query, {"entity_ids": entity_ids})

            # Fetch relationship info
            relation_results = self.db_query(relation_query, {"entity_ids": entity_ids})

            # Fetch chunk info via MENTIONS; fall back to text-match if empty
            chunk_results = self.db_query(chunk_query, {"entity_ids": entity_ids})
            _chunks_val = (chunk_results.iloc[0].get("chunks") if not chunk_results.empty else None) or []
            if not _chunks_val:
                chunk_results = self.db_query(chunk_fallback_query, {"entity_ids": entity_ids})

            self.performance_metrics["query_time"] += time.time() - query_start

            # Build result
            low_level = []

            # Add entity info
            if not entity_results.empty and 'entities' in entity_results.columns:
                entities = entity_results.iloc[0]['entities']
                if entities:
                    low_level.append("### Related Entities")
                    entity_dicts: List[Dict[str, Any]] = []
                    for entity in entities:
                        entity_desc = f"- **{entity['id']}** ({entity['type']}): {entity['description']}"
                        low_level.append(entity_desc)
                        entity_dicts.append(
                            {
                                "id": entity["id"],
                                "description": entity["description"],
                                "confidence": 0.65,
                                "type": entity["type"],
                            }
                        )
                    retrieval_results.extend(
                        results_from_entities(
                            entity_dicts,
                            source="hybrid_search",
                            confidence=0.65,
                        )
                    )

            # Add relationship info
            if not relation_results.empty and 'relationships' in relation_results.columns:
                relationships = relation_results.iloc[0]['relationships']
                if relationships:
                    low_level.append("\n### Entity Relationships")
                    relationship_dicts: List[Dict[str, Any]] = []
                    for rel in relationships:
                        rel_desc = f"- **{rel['start']}** -{rel['type']}-> **{rel['end']}**: {rel['description']}"
                        low_level.append(rel_desc)
                        relationship_dicts.append(
                            {
                                "start": rel["start"],
                                "end": rel["end"],
                                "type": rel["type"],
                                "description": rel.get("description", ""),
                                "confidence": 0.6,
                                "weight": 0.6,
                            }
                        )
                    retrieval_results.extend(
                        results_from_relationships(
                            relationship_dicts,
                            source="hybrid_search",
                            confidence=0.6,
                        )
                    )

            # Add chunk info
            if not chunk_results.empty and 'chunks' in chunk_results.columns:
                chunks = chunk_results.iloc[0]['chunks']
                if chunks:
                    low_level.append("\n### Relevant Text")
                    for chunk in chunks:
                        chunk_text = f"- ID: {chunk['id']}\n  Content: {chunk['text']}"
                        low_level.append(chunk_text)
                        retrieval_results.append(
                            create_retrieval_result(
                                evidence=chunk.get("text", ""),
                                source="hybrid_search",
                                granularity="Chunk",
                                metadata=create_retrieval_metadata(
                                    source_id=str(chunk.get("id")),
                                    source_type="chunk",
                                    confidence=0.7,
                                    extra={"raw_chunk": chunk},
                                ),
                                score=0.7,
                            )
                        )

            if not low_level:
                return "No relevant low-level content found.", retrieval_results

            return "\n".join(low_level), retrieval_results
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"Entity query failed: {e}")
            return "Error querying entity information.", retrieval_results

    # Words too generic or too short to usefully match XBRL CamelCase concept names.
    _XBRL_SKIP_WORDS = frozenset({
        "a", "an", "the", "and", "or", "of", "in", "to", "by", "for",
        "is", "are", "was", "its", "this", "that", "with", "from",
        "net", "total", "gross", "per",
    })

    def _retrieve_numeric_facts(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve FinancialFact and Table nodes relevant to the query.
        Returns a formatted string for inclusion in the LLM prompt.

        Matching strategy
        -----------------
        LLM-extracted keywords are natural-language phrases (e.g. "net revenue",
        "iPhone segment revenue").  XBRL concept names are CamelCase identifiers
        (e.g. "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax").
        A phrase-level CONTAINS check almost never matches CamelCase names.

        To fix this without adding a document-agnostic value fallback (which
        would inject Apple figures into a Walmart query), each multi-word phrase
        is also split into individual content words and those words are added as
        additional match terms.  "net revenue" → also tries "revenue", which DOES
        match the XBRL name.  The query remains keyword-scoped and document-safe.
        """
        if not keywords:
            keywords = query.split()

        # Build a de-duplicated list of search terms:
        #   1. Original keyword phrases (for context_ref / segment matching)
        #   2. Individual content words from each phrase (for CamelCase name matching)
        seen_terms: dict = {}
        for kw in keywords[:8]:
            seen_terms.setdefault(kw.lower(), kw)
            for word in kw.split():
                if len(word) > 3 and word.lower() not in self._XBRL_SKIP_WORDS:
                    seen_terms.setdefault(word.lower(), word)
        terms = list(seen_terms.values())[:14]  # cap to keep WHERE clause manageable

        keyword_conditions = []
        params: Dict[str, Any] = {"limit": 20}
        for i, term in enumerate(terms):
            p = f"kw{i}"
            params[p] = term
            keyword_conditions.append(
                f"(toLower(f.name) CONTAINS toLower(${p}) "
                f"OR toLower(f.context_ref) CONTAINS toLower(${p}) "
                f"OR toLower(coalesce(f.segment,'')) CONTAINS toLower(${p}))"
            )

        lines = []
        evidence: List[RetrievalResult] = []

        # Build RETURN clause from FINANCIAL_FACT_WRITE_FIELDS so it stays in sync with the model
        _fact_returns = ", ".join(f"f.{field} AS {field}" for field in FINANCIAL_FACT_WRITE_FIELDS)
        _fact_select = f"d.fileName AS doc, {_fact_returns}"

        # --- FinancialFact nodes ---
        if keyword_conditions:
            fact_cypher = (
                "MATCH (d:`__Document__`)-[:HAS_FACT]->(f:FinancialFact) "
                "WHERE " + " OR ".join(keyword_conditions) +
                f" RETURN {_fact_select} "
                "ORDER BY f.period_end DESC, f.value DESC LIMIT $limit"
            )
        else:
            fact_cypher = (
                "MATCH (d:`__Document__`)-[:HAS_FACT]->(f:FinancialFact) "
                f"RETURN {_fact_select} "
                "ORDER BY f.period_end DESC, f.value DESC LIMIT $limit"
            )

        try:
            fact_df = self.db_query(fact_cypher, params)
            if not fact_df.empty:
                lines.append("### Financial Facts (XBRL)")
                for _, row in fact_df.iterrows():
                    period = row.get("period_end") or ""
                    scale = row.get("scale")
                    scale_note = f", scale: 10^{scale}" if scale is not None else ""
                    lines.append(
                        f"- [{row['doc']}] {row['name']}: {row['value']} {row.get('unit','') or ''} "
                        f"(period: {period}, context: {row.get('context_ref','') or ''}{scale_note})"
                    )
                    fact_id = str(row.get("fact_id") or f"{row['name']}@{period}")
                    evidence.append(create_retrieval_result(
                        evidence=str(row.get("value", "")),
                        source="hybrid_search",
                        granularity="Chunk",
                        metadata=create_retrieval_metadata(
                            source_id=fact_id,
                            source_type="financial_fact",
                            confidence=0.9,
                            extra={"doc": row["doc"], "name": row.get("name"), "period_end": period},
                        ),
                        score=0.9,
                    ))
        except Exception as e:
            print(f"FinancialFact query failed: {e}")

        # --- Table nodes ---
        table_conditions = []
        table_params: Dict[str, Any] = {"limit": 5}
        for i, kw in enumerate(keywords[:5]):
            p = f"tkw{i}"
            table_params[p] = kw
            table_conditions.append(
                f"(toLower(t.content) CONTAINS toLower(${p})"
                f" OR toLower(coalesce(t.caption,'')) CONTAINS toLower(${p})"
                f" OR toLower(coalesce(t.section,'')) CONTAINS toLower(${p}))"
            )

        if table_conditions:
            table_cypher = (
                "MATCH (d:`__Document__`)-[:HAS_TABLE]->(t:Table) "
                "WHERE " + " OR ".join(table_conditions) +
                " RETURN d.fileName AS doc, t.table_id AS table_id, "
                "t.caption AS caption, t.section AS section, t.source AS source, "
                "t.content AS content "
                "LIMIT $limit"
            )
            try:
                table_df = self.db_query(table_cypher, table_params)
                if not table_df.empty:
                    lines.append("\n### Relevant Tables")
                    for _, row in table_df.iterrows():
                        caption = row.get("caption") or ""
                        section = row.get("section") or ""
                        source = row.get("source") or ""
                        lines.append(
                            f"- [{row['doc']}] Table {row['table_id']} {caption}"
                            + (f" | Section: {section}" if section else "")
                            + (f" | Source: {source}" if source else "")
                            + f":\n  {row['content']}"
                        )
                        evidence.append(create_retrieval_result(
                            evidence=str(row.get("content", "")),
                            source="hybrid_search",
                            granularity="Chunk",
                            metadata=create_retrieval_metadata(
                                source_id=f"{row['doc']}:table:{row['table_id']}",
                                source_type="financial_fact",
                                confidence=0.8,
                                extra={"doc": row["doc"], "table_id": row.get("table_id"), "caption": caption},
                            ),
                            score=0.8,
                        ))
            except Exception as e:
                print(f"Table query failed: {e}")

        return "\n".join(lines) if lines else "No structured numeric facts found.", evidence

    def _retrieve_high_level_content(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve high-level content (communities and thematic concepts).

        Args:
            query: Query string
            keywords: High-level keyword list

        Returns:
            Tuple[str, List[RetrievalResult]]: Formatted content and corresponding evidence
        """
        query_start = time.time()
        retrieval_results: List[RetrievalResult] = []

        # Build keyword conditions
        keyword_conditions = []
        params = {"level": self.community_level, "limit": self.top_communities}

        if keywords:
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                params[param_name] = keyword
                keyword_conditions.append(f"c.summary CONTAINS ${param_name} OR c.full_content CONTAINS ${param_name}")

        # Build community query
        community_query = """
        // Filter communities by keyword
        MATCH (c:__Community__ {level: $level})
        """

        if keyword_conditions:
            community_query += "WHERE " + " OR ".join(keyword_conditions)
        else:
            # If no keywords, use query text directly
            params["query"] = query
            community_query += "WHERE c.summary CONTAINS $query OR c.full_content CONTAINS $query"

        # Add ordering and limit
        community_query += """
        WITH c
        ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
        LIMIT $limit
        RETURN c.id AS id, c.summary AS summary, c.full_content AS full_content
        """

        try:
            community_results = self.db_query(community_query, params)

            self.performance_metrics["query_time"] += time.time() - query_start

            # Process results
            if community_results.empty:
                return "No relevant high-level content found.", retrieval_results

            # Build formatted high-level content
            high_level = ["### Related Thematic Concepts"]

            for _, row in community_results.iterrows():
                full_content = row.get("full_content") or ""
                community_desc = f"- **Community {row['id']}**:\n  {row['summary']}"
                if full_content:
                    community_desc += f"\n  Full Content: {full_content}"
                high_level.append(community_desc)
                retrieval_results.append(
                    create_retrieval_result(
                        evidence=row.get("summary", ""),
                        source="hybrid_search",
                        granularity="DO",
                        metadata=create_retrieval_metadata(
                            source_id=str(row.get("id")),
                            source_type="community",
                            confidence=0.6,
                            community_id=str(row.get("id")),
                            extra={"raw_community": row.to_dict()},
                        ),
                        score=0.6,
                    )
                )

            return "\n".join(high_level), retrieval_results
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"Community query failed: {e}")
            return "Error querying community information.", retrieval_results

    def _retrieve_filing_sections(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve FilingSection nodes that match query keywords.

        Args:
            query: Query string (used as fallback when no keywords)
            keywords: Keyword list for matching section title/content

        Returns:
            Tuple[str, List[RetrievalResult]]: Formatted content and evidence
        """
        params: Dict[str, Any] = {"limit": 5}
        keyword_conditions: List[str] = []

        if keywords:
            for i, kw in enumerate(keywords[:5]):
                p = f"keyword{i}"
                params[p] = kw
                keyword_conditions.append(
                    f"(toLower(s.title) CONTAINS toLower(${p})"
                    f" OR toLower(s.content) CONTAINS toLower(${p}))"
                )
        else:
            params["query"] = query
            keyword_conditions.append(
                "(toLower(s.title) CONTAINS toLower($query)"
                " OR toLower(s.content) CONTAINS toLower($query))"
            )

        section_cypher = (
            "MATCH (d:`__Document__`)-[:HAS_SECTION]->(s:FilingSection) "
            "WHERE " + " OR ".join(keyword_conditions) +
            " RETURN d.fileName AS doc, s.item AS item, s.title AS title, s.content AS content "
            "ORDER BY s.item "
            "LIMIT $limit"
        )

        sec_evidence: List[RetrievalResult] = []
        try:
            section_df = self.db_query(section_cypher, params)
            if section_df.empty:
                return "No relevant filing sections found.", sec_evidence

            lines = ["### Filing Sections (SEC Document Sections)"]
            for _, row in section_df.iterrows():
                item = row.get("item") or ""
                title = row.get("title") or ""
                content = row.get("content") or ""
                header = f"- [{row['doc']}]"
                if item:
                    header += f" Item {item}"
                if title:
                    header += f": {title}"
                lines.append(header)
                if content:
                    lines.append(f"  {content}")
                sec_evidence.append(create_retrieval_result(
                    evidence=content,
                    source="hybrid_search",
                    granularity="Chunk",
                    metadata=create_retrieval_metadata(
                        source_id=f"{row['doc']}:item:{item}" if item else f"{row['doc']}:{title}",
                        source_type="filing_section",
                        confidence=0.8,
                        extra={"doc": row["doc"], "item": item, "title": title},
                    ),
                    score=0.8,
                ))
            return "\n".join(lines), sec_evidence
        except Exception as e:
            print(f"FilingSection query failed: {e}")
            return "Error querying filing sections.", sec_evidence

    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        """
        Execute hybrid search and return a structured result containing evidence and answer.
        """
        overall_start = time.time()

        # Parse input
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            # Support directly passing categorized keywords
            low_keywords = query_input.get("low_level_keywords", [])
            high_keywords = query_input.get("high_level_keywords", [])
        else:
            query = str(query_input)
            # Extract keywords
            keywords = self.extract_keywords(query)
            low_keywords = keywords.get("low_level", [])
            high_keywords = keywords.get("high_level", [])

        # Check cache
        cache_key = query
        if low_keywords or high_keywords:
            cache_key = self.cache_manager.key_strategy.generate_key(
                query,
                low_level_keywords=low_keywords,
                high_level_keywords=high_keywords
            )

        cached_result = self.cache_manager.get(cache_key)
        if isinstance(cached_result, dict):
            return cached_result

        try:
            # 1. Retrieve low-level content (entities and relationships)
            low_level_content, low_evidence = self._retrieve_low_level_content(query, low_keywords)

            # 2. Retrieve high-level content (communities and themes)
            high_level_content, high_evidence = self._retrieve_high_level_content(query, high_keywords)

            # 3. Retrieve structured numeric facts (FinancialFact + Table nodes)
            all_keywords = list(dict.fromkeys(low_keywords + high_keywords))
            numeric_facts_content, numeric_evidence = self._retrieve_numeric_facts(query, all_keywords)

            # 4. Retrieve filing sections (FilingSection nodes)
            filing_sections_content, sections_evidence = self._retrieve_filing_sections(query, all_keywords)

            # 5. Generate final answer
            llm_start = time.time()

            answer = self.query_chain.invoke({
                "query": query,
                "low_level": low_level_content,
                "high_level": high_level_content,
                "numeric_facts": numeric_facts_content,
                "filing_sections": filing_sections_content,
                "response_type": response_type
            })

            self.performance_metrics["llm_time"] += time.time() - llm_start

            all_evidence = merge_retrieval_results(low_evidence, high_evidence, numeric_evidence, sections_evidence)

            # Build real citation from actual retrieved IDs — override whatever the LLM wrote
            def _fmt(ids: list, quote: bool = False) -> str:
                if not ids:
                    return "[]"
                return "[" + ", ".join(f"'{i}'" if quote else str(i) for i in ids) + "]"

            entity_ids_cited  = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "entity"         and r.metadata.source_id][:5]
            rel_ids_cited     = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "relationship"   and r.metadata.source_id][:5]
            chunk_ids_cited   = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "chunk"          and r.metadata.source_id][:5]
            report_ids_cited  = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "community"      and r.metadata.source_id][:5]
            fact_ids_cited    = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "financial_fact" and r.metadata.source_id][:5]
            section_ids_cited = [r.metadata.source_id for r in all_evidence if r.metadata.source_type == "filing_section" and r.metadata.source_id][:5]

            citation_block = (
                "\n\n### Citation Data\n"
                f"{{'data': {{"
                f"'Entities':{_fmt(entity_ids_cited)}, "
                f"'Reports':{_fmt(report_ids_cited)}, "
                f"'Relationships':{_fmt(rel_ids_cited)}, "
                f"'Chunks':{_fmt(chunk_ids_cited, quote=True)}, "
                f"'Facts':{_fmt(fact_ids_cited, quote=True)}, "
                f"'Sections':{_fmt(section_ids_cited, quote=True)}"
                f"}} }}"
            )

            if answer:
                # Replace any LLM-generated citation block
                answer = re.sub(r"\n*###\s*Citation Data\s*\n.*", citation_block, answer, flags=re.DOTALL)
                if "### Citation Data" not in answer:
                    answer += citation_block

            structured_result = {
                "query": query,
                "low_level_content": low_level_content,
                "high_level_content": high_level_content,
                "filing_sections_content": filing_sections_content,
                "final_answer": answer if answer else "No relevant information found.",
                "retrieval_results": results_to_payload(all_evidence),
            }

            # Cache result
            self.cache_manager.set(
                cache_key,
                structured_result,
                low_level_keywords=low_keywords,
                high_level_keywords=high_keywords
            )

            self.performance_metrics["total_time"] = time.time() - overall_start

            return structured_result

        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            print(error_msg)
            return {
                "query": query,
                "low_level_content": "",
                "high_level_content": "",
                "final_answer": error_msg,
                "retrieval_results": [],
                "error": error_msg,
            }

    def search(self, query_input: Any) -> str:
        """
        Execute hybrid search combining low-level and high-level content.

        Args:
            query_input: String query or dict containing query and keywords

        Returns:
            str: Generated final answer
        """
        structured = self.structured_search(query_input)
        return structured.get("final_answer", "No relevant information found.")

    def get_global_tool(self) -> BaseTool:
        """
        Get the global search tool instance.

        Returns:
            BaseTool: Global search tool
        """
        class GlobalSearchTool(BaseTool):
            name: str = "global_retriever"
            description: str = gl_description

            def _run(self_tool, query: Any) -> str:
                # Use only high-level content
                if isinstance(query, dict) and "query" in query:
                    original_query = query["query"]
                    keywords = query.get("keywords", [])
                    query = {
                        "query": original_query,
                        "high_level_keywords": keywords,
                        "low_level_keywords": []  # skip low-level keywords
                    }
                else:
                    # Extract keywords
                    keywords = self.extract_keywords(str(query))
                    query = {
                        "query": str(query),
                        "high_level_keywords": keywords.get("high_level", []),
                        "low_level_keywords": []
                    }

                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("Async execution not implemented")

        return GlobalSearchTool()

    def close(self):
        """Close resources."""
        super().close()
