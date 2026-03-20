from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
# [GDS] from graphdatascience import GraphDataScience  # GDS - commented out for Aura Free compatibility
from langchain_community.graphs import Neo4jGraph
import psutil
import os
import time
from contextlib import contextmanager

from graphrag_agent.config.settings import MAX_WORKERS, GDS_CONCURRENCY, GDS_MEMORY_LIMIT

class BaseCommunityDetector(ABC):
    """Base class for community detection."""

    def __init__(self, graph: Neo4jGraph, gds=None):
        # [GDS] gds: GraphDataScience — re-enable and swap arg order when switching back to GDS
        # [GDS] self.gds = gds
        self.graph = graph
        # [GDS] self.projection_name = "communities"
        # [GDS] self.G = None

        # Performance statistics
        self.projection_time = 0
        self.detection_time = 0
        self.save_time = 0

        # System resources
        self._init_system_resources()

    def _init_system_resources(self):
        """Initialize system resource parameters."""
        self.memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.cpu_count = os.cpu_count() or 4
        self._adjust_parameters()

    def _adjust_parameters(self):
        """Adjust runtime parameters."""
        # Set concurrency
        self.max_concurrency = GDS_CONCURRENCY

        # Use configured memory limit
        memory_gb = GDS_MEMORY_LIMIT

        # Adjust limits based on configured memory size
        if memory_gb > 32:
            self.node_count_limit = 100000
            self.timeout_seconds = 600
        elif memory_gb > 16:
            self.node_count_limit = 50000
            self.timeout_seconds = 300
        else:
            self.node_count_limit = 20000
            self.timeout_seconds = 180

        print(f"Community detection parameters: CPU={MAX_WORKERS}, memory={memory_gb:.1f}GB, "
            f"concurrency={self.max_concurrency}, node limit={self.node_count_limit}")

    # [GDS] @contextmanager
    # [GDS] def _graph_projection_context(self):
    # [GDS]     """Context manager for graph projection."""
    # [GDS]     try:
    # [GDS]         projection_start = time.time()
    # [GDS]         self.create_projection()
    # [GDS]         self.projection_time = time.time() - projection_start
    # [GDS]         yield
    # [GDS]     finally:
    # [GDS]         cleanup_start = time.time()
    # [GDS]         self.cleanup()
    # [GDS]         print(f"Graph projection cleanup complete in {time.time() - cleanup_start:.2f}s")

    def process(self) -> Dict[str, Any]:
        """Execute the full community detection pipeline."""
        start_time = time.time()
        print(f"Starting {self.__class__.__name__} community detection...")

        results = {
            'status': 'success',
            'algorithm': self.__class__.__name__,
            'details': {}
        }

        try:
            # [GDS] with self._graph_projection_context():  # GDS projection - commented out for Aura Free
            # Pure Cypher: run detection directly without graph projection
            detection_start = time.time()
            detection_result = self.detect_communities()
            self.detection_time = time.time() - detection_start
            results['details']['detection'] = detection_result

            save_start = time.time()
            save_result = self.save_communities()
            self.save_time = time.time() - save_start
            results['details']['save'] = save_result

            # Add performance statistics
            total_time = time.time() - start_time
            results['performance'] = {
                'totalTime': total_time,
                'projectionTime': self.projection_time,
                'detectionTime': self.detection_time,
                'saveTime': self.save_time
            }

            return results

        except Exception as e:
            results.update({
                'status': 'error',
                'error': str(e),
                'elapsed': time.time() - start_time
            })
            raise

    # [GDS] def create_projection(self) -> Tuple[Any, Dict]:
    # [GDS]     """Create graph projection."""
    # [GDS]     pass

    @abstractmethod
    def detect_communities(self) -> Dict[str, Any]:
        """Detect communities."""
        pass

    @abstractmethod
    def save_communities(self) -> Dict[str, int]:
        """Save community results."""
        pass

    # [GDS] def cleanup(self):
    # [GDS]     """Clean up in-memory GDS projection graph."""
    # [GDS]     if self.G:
    # [GDS]         try:
    # [GDS]             self.G.drop()
    # [GDS]             print("Community projection graph cleaned up")
    # [GDS]         except Exception as e:
    # [GDS]             print(f"Error cleaning up projection graph: {e}")
    # [GDS]         finally:
    # [GDS]             self.G = None
