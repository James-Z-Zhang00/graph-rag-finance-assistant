from langchain_community.graphs import Neo4jGraph
# [GDS] from graphdatascience import GraphDataScience  # GDS - commented out for Aura Free compatibility
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
    def create(cls, algorithm: str, graph: Neo4jGraph, gds=None) -> BaseCommunityDetector:
        # [GDS] gds: GraphDataScience — re-enable when switching back to GDS
        algorithm = algorithm.lower()
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return cls.ALGORITHMS[algorithm](graph, gds)

__all__ = ['CommunityDetectorFactory', 'BaseCommunityDetector',
           'LeidenDetector', 'SLLPADetector']
