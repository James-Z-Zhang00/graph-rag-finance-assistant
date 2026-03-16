"""
Retrieval Result Module

Defines the unified RetrievalResult interface, aligned with IoD-style
multi-granularity retrieval.
"""

from typing import Union, Dict, Any, Optional, Literal, Tuple
from datetime import datetime
import uuid

from pydantic import BaseModel, Field

RETRIEVAL_SOURCE_CHOICES: Tuple[str, ...] = (
    "local_search",
    "global_search",
    "hybrid_search",
    "naive_search",
    "deep_research",
    "deeper_research",
    "chain_exploration",
    "graph_search",
    "hybrid",
    "custom",
)

RetrievalSourceLiteral = Literal[
    "local_search",
    "global_search",
    "hybrid_search",
    "naive_search",
    "deep_research",
    "deeper_research",
    "chain_exploration",
    "graph_search",
    "hybrid",
    "custom",
]


class RetrievalMetadata(BaseModel):
    """
    Retrieval metadata

    Records the source, confidence, and granularity level of a retrieval result.
    """
    # Data source ID (DO ID or Chunk ID, traceable back to the original data source)
    source_id: str = Field(description="Unique identifier of the data source")

    # Data source type
    source_type: Literal[
        "document",
        "chunk",
        "entity",
        "relationship",
        "community",
        "subgraph",
        "financial_fact",
        "filing_section",
    ] = Field(description="Type of data source")

    # Confidence score (0.0–1.0)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score of the retrieval result"
    )

    # Data timestamp (used to filter stale data)
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the data"
    )

    # DO level (Digital Object level)
    do_level: Optional[Literal["L0-DO", "L1-DO", "L2-DO"]] = Field(
        default=None,
        description="Digital Object level; only applicable at DO granularity"
    )

    # Community ID (applicable during LocalSearch)
    community_id: Optional[str] = Field(
        default=None,
        description="ID of the community this result belongs to"
    )

    # Graph traversal hop count (applicable during GraphSearch)
    hop_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of hops in the graph traversal"
    )

    # Additional metadata (for extensibility)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata"
    )


class RetrievalResult(BaseModel):
    """
    Unified retrieval result interface

    Adapts IoD-style multi-granularity retrieval
    (DO / Chunk / AtomicKnowledge / KGSubgraph).
    """
    # Unique result identifier
    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the retrieval result"
    )

    # Retrieval granularity
    granularity: Literal[
        "DO",               # Digital Object (document level)
        "L2-DO",            # Level 2 Digital Object (paragraph level)
        "Chunk",            # Text chunk
        "AtomicKnowledge",  # Atomic knowledge (triples, etc.)
        "KGSubgraph"        # Knowledge graph subgraph
    ] = Field(description="Retrieval granularity")

    # Retrieved content (text or structured data)
    evidence: Union[str, Dict[str, Any]] = Field(
        description="Retrieved content; may be plain text or structured data"
    )

    # Metadata
    metadata: RetrievalMetadata = Field(description="Metadata for the retrieval result")

    # Retrieval source
    source: RetrievalSourceLiteral = Field(description="Tool that performed the retrieval")

    # Similarity / relevance score (0.0–1.0)
    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity or relevance score"
    )

    # Creation timestamp
    created_at: datetime = Field(default_factory=datetime.now)

    def get_citation(self, format_type: str = "default") -> str:
        """
        Generate a formatted citation string.

        Args:
            format_type: Citation format type (default | apa | mla)

        Returns:
            Formatted citation string
        """
        if format_type == "apa":
            # APA format example
            return f"[{self.result_id[:8]}] ({self.metadata.timestamp.year}). {self.metadata.source_type}. Retrieved from {self.metadata.source_id}"
        elif format_type == "mla":
            # MLA format example
            return f'[{self.result_id[:8]}] "{self.metadata.source_type}." {self.metadata.source_id}, {self.metadata.timestamp.year}.'
        else:
            # Default format
            source_desc = f"{self.metadata.source_type}:{self.metadata.source_id}"
            if self.metadata.community_id:
                source_desc += f" (community:{self.metadata.community_id})"
            return f"[{self.result_id[:8]}] Source: {source_desc} (confidence:{self.metadata.confidence:.2f})"

    @classmethod
    def merge(cls, results: list["RetrievalResult"]) -> "RetrievalResult":
        """
        Merge multiple retrieval results sharing the same source_id.

        Selects the result with the highest score as the merged output.

        Args:
            results: List of results to merge

        Returns:
            Single merged result
        """
        if not results:
            raise ValueError("Cannot merge an empty result list")

        # Sort by score descending, take the top result
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        best_result = sorted_results[0]

        # Optionally merge evidence — here we retain the evidence from the highest-scoring result
        return best_result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "result_id": self.result_id,
            "granularity": self.granularity,
            "evidence": self.evidence,
            "metadata": self.metadata.model_dump(),
            "source": self.source,
            "score": self.score,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create from dictionary format"""
        metadata_data = data.get("metadata", {})
        metadata = RetrievalMetadata(**metadata_data)

        created_at_value = data.get("created_at")
        if created_at_value:
            # Handle ISO format strings ending with Z or missing UTC offset
            cleaned_value = created_at_value.replace("Z", "+00:00")
            try:
                created_at = datetime.fromisoformat(cleaned_value)
            except ValueError:
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        return cls(
            result_id=data.get("result_id", str(uuid.uuid4())),
            granularity=data["granularity"],
            evidence=data["evidence"],
            metadata=metadata,
            source=data["source"],
            score=data.get("score", 0.5),
            created_at=created_at
        )
