from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from .base import BaseCommunityDetector
from .leiden import LeidenDetector
from .sllpa import SLLPADetector

class CommunityDetectorFactory:
    """Community detector factory class."""

    ALGORITHMS = {
        'leiden': LeidenDetector,
        'sllpa': SLLPADetector
    }

    @classmethod
    def create(cls, algorithm: str, gds: GraphDataScience, graph: Neo4jGraph) -> BaseCommunityDetector:
        algorithm = algorithm.lower()
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return cls.ALGORITHMS[algorithm](gds, graph)

__all__ = ['CommunityDetectorFactory', 'BaseCommunityDetector',
           'LeidenDetector', 'SLLPADetector']
