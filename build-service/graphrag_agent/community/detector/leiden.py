from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from graphrag_agent.config.settings import GDS_CONCURRENCY

class LeidenDetector(GraphProjectionMixin, BaseCommunityDetector):
    """Community detection implementation using the Leiden algorithm."""

    def detect_communities(self) -> Dict[str, Any]:
        """Run Leiden algorithm community detection."""
        if not self.G:
            raise ValueError("Please create the graph projection first")

        print("Starting Leiden community detection...")

        try:
            # Check connected components
            wcc = self.gds.wcc.stats(self.G)
            print(f"Graph contains {wcc.get('componentCount', 0)} connected components")

            # Execute Leiden algorithm
            result = self.gds.leiden.write(
                self.G,
                writeProperty="communities",
                includeIntermediateCommunities=True,
                relationshipWeightProperty="weight",
                **self._get_optimized_leiden_params()
            )

            return {
                'componentCount': wcc.get('componentCount', 0),
                'componentDistribution': wcc.get('componentDistribution', {}),
                'communityCount': result.get('communityCount', 0),
                'modularity': result.get('modularity', 0),
                'ranLevels': result.get('ranLevels', 0)
            }

        except Exception as e:
            print(f"Leiden algorithm failed: {e}")
            return self._execute_fallback_leiden()

    def _execute_fallback_leiden(self) -> Dict[str, Any]:
        """Execute fallback Leiden algorithm."""
        print("Trying with fallback parameters...")

        try:
            result = self.gds.leiden.write(
                self.G,
                writeProperty="communities",
                includeIntermediateCommunities=False,
                gamma=0.5,
                tolerance=0.001,
                maxLevels=2,
                concurrency=1
            )

            return {
                'communityCount': result.get('communityCount', 0),
                'modularity': result.get('modularity', 0),
                'ranLevels': result.get('ranLevels', 0),
                'note': 'Used fallback parameters'
            }
        except Exception as e:
            raise ValueError(f"Leiden algorithm failed: {e}")

    def _get_optimized_leiden_params(self) -> Dict[str, Any]:
        """Get optimized Leiden algorithm parameters."""
        if self.memory_mb > 32 * 1024:  # >32GB
            return {
                'gamma': 1.0,
                'tolerance': 0.0001,
                'maxLevels': 10,
                'concurrency': GDS_CONCURRENCY
            }
        elif self.memory_mb > 16 * 1024:  # >16GB
            return {
                'gamma': 1.0,
                'tolerance': 0.0005,
                'maxLevels': 5,
                'concurrency': max(1, GDS_CONCURRENCY - 1)
            }
        else:  # Low-memory system
            return {
                'gamma': 0.8,
                'tolerance': 0.001,
                'maxLevels': 3,
                'concurrency': max(1, GDS_CONCURRENCY // 2)
            }

    def save_communities(self) -> Dict[str, int]:
        """Save community detection results from the Leiden algorithm."""
        print("Saving Leiden community detection results...")

        try:
            # Create constraint
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )

            # Save base-level community relationships
            base_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 0
            WITH collect({entityId: id(e), community: e.communities[0]}) AS data
            UNWIND data AS item
            MERGE (c:`__Community__` {id: '0-' + toString(item.community)})
            ON CREATE SET c.level = 0
            WITH item, c
            MATCH (e) WHERE id(e) = item.entityId
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) AS base_count
            """)

            base_count = base_result[0]['base_count'] if base_result else 0

            # Save higher-level community relationships
            higher_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 1
            WITH e, e.communities AS communities
            UNWIND range(1, size(communities) - 1) AS index
            WITH e, index, communities[index] AS current_community,
                 communities[index-1] AS previous_community

            MERGE (current:`__Community__` {id: toString(index) + '-' +
                                              toString(current_community)})
            ON CREATE SET current.level = index

            WITH e, current, previous_community, index
            MATCH (previous:`__Community__` {id: toString(index - 1) + '-' +
                                              toString(previous_community)})
            MERGE (previous)-[:IN_COMMUNITY]->(current)

            RETURN count(*) AS higher_count
            """)

            higher_count = higher_result[0]['higher_count'] if higher_result else 0

            return {'saved_communities': base_count + higher_count}

        except Exception as e:
            print(f"Community save failed: {e}")
            return self._save_communities_fallback()
