from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from graphrag_agent.config.settings import GDS_CONCURRENCY

class LeidenDetector(GraphProjectionMixin, BaseCommunityDetector):
    """Community detection implementation using the Leiden algorithm."""

    def detect_communities(self) -> Dict[str, Any]:
        """Run Label Propagation Algorithm for community detection (pure Cypher, Aura-compatible).
        Replaces gds.leiden.write() which requires GDS unavailable on AuraDB Free.
        Result is written to e.communities as a list for save_communities() compatibility.
        """
        print("Starting Label Propagation community detection (pure Cypher)...")

        # Initialize: each entity gets its own unique community ID
        self.graph.query("""
            MATCH (e:`__Entity__`)
            SET e.community_lpa = id(e)
        """)

        # Iteratively propagate: each node adopts the most common community among its neighbours
        max_iterations = 10
        for i in range(max_iterations):
            result = self.graph.query("""
                MATCH (e:`__Entity__`)
                OPTIONAL MATCH (e)--(n:`__Entity__`)
                WITH e, n.community_lpa AS lbl
                WHERE lbl IS NOT NULL
                WITH e, lbl, count(*) AS freq
                ORDER BY freq DESC, lbl ASC
                WITH e, collect(lbl)[0] AS best_label
                WHERE best_label IS NOT NULL AND best_label <> e.community_lpa
                SET e.community_lpa = best_label
                RETURN count(*) AS updates
            """)
            updates = result[0]['updates'] if result else 0
            print(f"  LPA iteration {i + 1}: {updates} updates")
            if updates == 0:
                print(f"  Converged after {i + 1} iterations")
                break

        # Convert to list format — save_communities() reads e.communities[0] for level 0
        self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.community_lpa IS NOT NULL
            SET e.communities = [e.community_lpa]
            REMOVE e.community_lpa
        """)

        result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL
            RETURN count(DISTINCT e.communities[0]) AS communityCount
        """)
        community_count = result[0]['communityCount'] if result else 0
        print(f"LPA complete: {community_count} communities detected")

        return {
            'communityCount': community_count,
            'componentCount': community_count,
            'modularity': 0.0,
            'ranLevels': 1
        }

    # [GDS] def detect_communities_gds(self) -> Dict[str, Any]:
    # [GDS]     """Run Leiden algorithm community detection (requires GDS)."""
    # [GDS]     if not self.G:
    # [GDS]         raise ValueError("Please create the graph projection first")
    # [GDS]     print("Starting Leiden community detection...")
    # [GDS]     try:
    # [GDS]         wcc = self.gds.wcc.stats(self.G)
    # [GDS]         print(f"Graph contains {wcc.get('componentCount', 0)} connected components")
    # [GDS]         result = self.gds.leiden.write(
    # [GDS]             self.G,
    # [GDS]             writeProperty="communities",
    # [GDS]             includeIntermediateCommunities=True,
    # [GDS]             relationshipWeightProperty="weight",
    # [GDS]             **self._get_optimized_leiden_params()
    # [GDS]         )
    # [GDS]         return {
    # [GDS]             'componentCount': wcc.get('componentCount', 0),
    # [GDS]             'componentDistribution': wcc.get('componentDistribution', {}),
    # [GDS]             'communityCount': result.get('communityCount', 0),
    # [GDS]             'modularity': result.get('modularity', 0),
    # [GDS]             'ranLevels': result.get('ranLevels', 0)
    # [GDS]         }
    # [GDS]     except Exception as e:
    # [GDS]         print(f"Leiden algorithm failed: {e}")
    # [GDS]         return self._execute_fallback_leiden()

    # [GDS] def _execute_fallback_leiden(self) -> Dict[str, Any]:
    # [GDS]     print("Trying with fallback parameters...")
    # [GDS]     try:
    # [GDS]         result = self.gds.leiden.write(
    # [GDS]             self.G,
    # [GDS]             writeProperty="communities",
    # [GDS]             includeIntermediateCommunities=False,
    # [GDS]             gamma=0.5,
    # [GDS]             tolerance=0.001,
    # [GDS]             maxLevels=2,
    # [GDS]             concurrency=1
    # [GDS]         )
    # [GDS]         return {
    # [GDS]             'communityCount': result.get('communityCount', 0),
    # [GDS]             'modularity': result.get('modularity', 0),
    # [GDS]             'ranLevels': result.get('ranLevels', 0),
    # [GDS]             'note': 'Used fallback parameters'
    # [GDS]         }
    # [GDS]     except Exception as e:
    # [GDS]         raise ValueError(f"Leiden algorithm failed: {e}")

    # [GDS] def _get_optimized_leiden_params(self) -> Dict[str, Any]:
    # [GDS]     if self.memory_mb > 32 * 1024:
    # [GDS]         return {'gamma': 1.0, 'tolerance': 0.0001, 'maxLevels': 10, 'concurrency': GDS_CONCURRENCY}
    # [GDS]     elif self.memory_mb > 16 * 1024:
    # [GDS]         return {'gamma': 1.0, 'tolerance': 0.0005, 'maxLevels': 5, 'concurrency': max(1, GDS_CONCURRENCY - 1)}
    # [GDS]     else:
    # [GDS]         return {'gamma': 0.8, 'tolerance': 0.001, 'maxLevels': 3, 'concurrency': max(1, GDS_CONCURRENCY // 2)}

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
