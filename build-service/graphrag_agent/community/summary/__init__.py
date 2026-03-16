from langchain_community.graphs import Neo4jGraph
from typing import Union
from .base import BaseSummarizer
from .leiden import LeidenSummarizer
from .sllpa import SLLPASummarizer

class CommunitySummarizerFactory:
    """Community summarizer factory class."""

    SUMMARIZERS = {
        'leiden': LeidenSummarizer,
        'sllpa': SLLPASummarizer
    }

    @classmethod
    def create_summarizer(cls,
                         algorithm: str,
                         graph: Neo4jGraph) -> BaseSummarizer:
        """
        Create a community summarizer instance.

        Args:
            algorithm: Algorithm type ('leiden' or 'sllpa')
            graph: Neo4j graph instance

        Returns:
            BaseSummarizer: Summarizer instance
        """
        algorithm = algorithm.lower()
        if algorithm not in cls.SUMMARIZERS:
            raise ValueError(f"Unsupported algorithm type: {algorithm}")

        summarizer_class = cls.SUMMARIZERS[algorithm]
        return summarizer_class(graph)

__all__ = ['CommunitySummarizerFactory', 'BaseSummarizer',
           'LeidenSummarizer', 'SLLPASummarizer']
