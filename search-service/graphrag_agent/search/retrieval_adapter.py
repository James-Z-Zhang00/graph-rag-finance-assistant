"""
Retrieval Result Adapter

Converts raw output from different search tools into the unified RetrievalResult
data model for consumption by the multi-agent pipeline.
"""
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import uuid

from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)


def create_retrieval_metadata(
    *,
    source_id: str,
    source_type: str,
    confidence: float = 0.5,
    timestamp: Optional[datetime] = None,
    do_level: Optional[str] = None,
    community_id: Optional[str] = None,
    hop_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RetrievalMetadata:
    """Build a unified RetrievalMetadata object."""
    return RetrievalMetadata(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        confidence=confidence,
        timestamp=timestamp or datetime.now(),
        do_level=do_level,  # type: ignore[arg-type]
        community_id=community_id,
        hop_count=hop_count,
        extra=extra or {},
    )


def create_retrieval_result(
    *,
    evidence: Any,
    source: str,
    granularity: str,
    metadata: RetrievalMetadata,
    score: float = 0.5,
    result_id: Optional[str] = None,
) -> RetrievalResult:
    """Build a RetrievalResult object."""
    return RetrievalResult(
        result_id=result_id or str(uuid.uuid4()),
        granularity=granularity,  # type: ignore[arg-type]
        evidence=evidence,
        metadata=metadata,
        source=source,  # type: ignore[arg-type]
        score=score,
    )


def results_to_payload(results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
    """Convert a list of RetrievalResults to a serializable list of dicts."""
    return [result.to_dict() for result in results]


def results_from_documents(
    docs: Iterable[Any],
    *,
    source: str,
    default_confidence: float = 0.6,
    granularity: str = "Chunk",
) -> List[RetrievalResult]:
    """
    Generate a list of RetrievalResults from LangChain Documents or similar objects.

    Args:
        docs: Iterator of LangChain Document or compatible objects
        source: Retrieval source (local_search / global_search / etc.)
        default_confidence: Default confidence score
        granularity: Result granularity
    """
    results: List[RetrievalResult] = []
    for doc in docs:
        if isinstance(doc, dict):
            metadata_dict = doc.get("metadata", {}) or {}
            score_value = doc.get("score", metadata_dict.get("score", default_confidence))
            page_content = doc.get("page_content") or doc.get("text") or metadata_dict.get("text") or ""
        else:
            metadata_dict = getattr(doc, "metadata", {}) or {}
            score_value = getattr(doc, "score", metadata_dict.get("score", default_confidence))
            page_content = getattr(doc, "page_content", None) or metadata_dict.get("text") or ""

        source_id = str(
            metadata_dict.get("id")
            or metadata_dict.get("source_id")
            or metadata_dict.get("chunk_id")
            or uuid.uuid4()
        )
        community_id = metadata_dict.get("community_id") or metadata_dict.get("community")
        score = float(score_value or default_confidence)
        metadata = create_retrieval_metadata(
            source_id=source_id,
            source_type="chunk",
            confidence=metadata_dict.get("confidence", score),
            community_id=community_id,
            extra={
                "source": metadata_dict.get("source"),
                "document_id": metadata_dict.get("document_id"),
                "raw_metadata": metadata_dict,
            },
        )
        evidence = page_content
        results.append(
            create_retrieval_result(
                evidence=evidence,
                source=source,
                granularity=granularity,
                metadata=metadata,
                score=score,
            )
        )
    return results


def results_from_entities(
    entities: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.55,
) -> List[RetrievalResult]:
    """Build RetrievalResults from a list of entities."""
    results: List[RetrievalResult] = []
    for entity in entities:
        entity_id = str(entity.get("id") or uuid.uuid4())
        description = entity.get("description") or entity.get("text") or ""
        metadata = create_retrieval_metadata(
            source_id=entity_id,
            source_type="entity",
            confidence=entity.get("confidence", confidence),
            extra={"raw_entity": entity},
        )
        results.append(
            create_retrieval_result(
                evidence=description,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=entity.get("weight", confidence),
            )
        )
    return results


def results_from_relationships(
    relationships: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.5,
) -> List[RetrievalResult]:
    """Generate RetrievalResults from relationship data."""
    results: List[RetrievalResult] = []
    for relation in relationships:
        relation_id = str(
            relation.get("id")
            or f"{relation.get('start')}->{relation.get('end')}:{relation.get('type')}"
            or uuid.uuid4()
        )
        description = relation.get("description") or ""
        metadata = create_retrieval_metadata(
            source_id=relation_id,
            source_type="relationship",
            confidence=relation.get("confidence", confidence),
            extra={"raw_relationship": relation},
        )
        evidence = description or f"{relation.get('start')} -{relation.get('type')}-> {relation.get('end')}"
        # Normalize weight to [0, 1] to avoid exceeding the score validation range
        raw_weight = relation.get("weight", confidence)
        normalized_score = min(1.0, max(0.0, raw_weight / 10.0 if raw_weight > 1.0 else raw_weight))
        results.append(
            create_retrieval_result(
                evidence=evidence,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=normalized_score,
            )
        )
    return results


def merge_retrieval_results(*result_groups: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    """Merge multiple RetrievalResult sequences and deduplicate by (source_id, granularity)."""
    merged: Dict[tuple[str, str], RetrievalResult] = {}
    for group in result_groups:
        for result in group:
            key = (result.metadata.source_id, result.granularity)
            existing = merged.get(key)
            if existing is None or result.score > existing.score:
                merged[key] = result
    return list(merged.values())
