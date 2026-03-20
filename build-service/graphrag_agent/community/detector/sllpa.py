from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from graphrag_agent.config.settings import GDS_CONCURRENCY

class SLLPADetector(GraphProjectionMixin, BaseCommunityDetector):
    """Community detection implementation using the SLLPA algorithm."""

    def detect_communities(self) -> Dict[str, Any]:
        """Run Label Propagation Algorithm for community detection (pure Cypher, Aura-compatible).
        Replaces gds.sllpa.write() which requires GDS unavailable on AuraDB Free.
        Result is written to e.communityIds as a list for save_communities() compatibility.
        """
        print("Starting Label Propagation community detection (pure Cypher, SLLPA replacement)...")

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

        # Convert to list format — save_communities() reads e.communityIds
        self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.community_lpa IS NOT NULL
            SET e.communityIds = [e.community_lpa]
            REMOVE e.community_lpa
        """)

        result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL
            RETURN count(DISTINCT e.communityIds[0]) AS communityCount
        """)
        community_count = result[0]['communityCount'] if result else 0
        print(f"LPA complete: {community_count} communities detected")

        return {
            'communityCount': community_count,
            'iterations': max_iterations
        }

    # [GDS] def detect_communities_gds(self) -> Dict[str, Any]:
    # [GDS]     """Run SLLPA algorithm community detection (requires GDS)."""
    # [GDS]     if not self.G:
    # [GDS]         raise ValueError("Please create the graph projection first")
    # [GDS]     print("Starting SLLPA community detection...")
    # [GDS]     try:
    # [GDS]         result = self.gds.sllpa.write(
    # [GDS]             self.G,
    # [GDS]             writeProperty="communityIds",
    # [GDS]             **self._get_optimized_sllpa_params()
    # [GDS]         )
    # [GDS]         community_count = result.get('communityCount', 0)
    # [GDS]         iterations = result.get('iterations', 0)
    # [GDS]         print(f"SLLPA complete: {community_count} communities, {iterations} iterations")
    # [GDS]         return {'communityCount': community_count, 'iterations': iterations}
    # [GDS]     except Exception as e:
    # [GDS]         print(f"SLLPA algorithm failed: {e}")
    # [GDS]         return self._execute_fallback_sllpa()

    # [GDS] def _execute_fallback_sllpa(self) -> Dict[str, Any]:
    # [GDS]     print("Trying with fallback parameters...")
    # [GDS]     try:
    # [GDS]         result = self.gds.sllpa.write(
    # [GDS]             self.G,
    # [GDS]             writeProperty="communityIds",
    # [GDS]             maxIterations=50,
    # [GDS]             minAssociationStrength=0.2,
    # [GDS]             concurrency=1
    # [GDS]         )
    # [GDS]         return {
    # [GDS]             'communityCount': result.get('communityCount', 0),
    # [GDS]             'iterations': result.get('iterations', 0),
    # [GDS]             'note': 'Used fallback parameters'
    # [GDS]         }
    # [GDS]     except Exception as e:
    # [GDS]         raise ValueError(f"SLLPA algorithm failed: {e}")

    # [GDS] def _get_optimized_sllpa_params(self) -> Dict[str, Any]:
    # [GDS]     if self.memory_mb > 32 * 1024:
    # [GDS]         return {'maxIterations': 100, 'minAssociationStrength': 0.05, 'concurrency': GDS_CONCURRENCY}
    # [GDS]     elif self.memory_mb > 16 * 1024:
    # [GDS]         return {'maxIterations': 80, 'minAssociationStrength': 0.08, 'concurrency': max(1, GDS_CONCURRENCY - 1)}
    # [GDS]     else:
    # [GDS]         return {'maxIterations': 50, 'minAssociationStrength': 0.1, 'concurrency': max(1, GDS_CONCURRENCY // 2)}

    def save_communities(self) -> Dict[str, int]:
        """Save SLLPA algorithm results."""
        print("Saving SLLPA community detection results...")

        try:
            # Create constraint
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )

            # Save communities
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL
            WITH count(e) AS entities_with_communities

            CALL {
                WITH entities_with_communities
                MATCH (e:`__Entity__`)
                WHERE e.communityIds IS NOT NULL
                WITH collect(e) AS entities
                CALL {
                    WITH entities
                    UNWIND entities AS e
                    UNWIND range(0, size(e.communityIds) - 1, 1) AS index
                    MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
                    ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                }
                RETURN count(*) AS processed_count
            }

            RETURN CASE
                WHEN entities_with_communities > 0 THEN entities_with_communities
                ELSE 0
            END AS total_count
            """)

            total_count = result[0]['total_count'] if result else 0
            print(f"Saved {total_count} SLLPA community relationships")

            return {'saved_communities': total_count}

        except Exception as e:
            print(f"Failed to save SLLPA community results: {e}")
            return self._save_communities_fallback()

    def _save_communities_fallback(self) -> Dict[str, int]:
        """Fallback community save method."""
        print("Trying simplified community save method...")

        try:
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL AND size(e.communityIds) > 0
            WITH e, e.communityIds[0] AS primary_community
            MERGE (c:`__Community__` {id: '0-' + toString(primary_community)})
            ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) as count
            """)

            count = result[0]['count'] if result else 0
            print(f"Saved {count} community relationships using simplified method")

            return {
                'saved_communities': count,
                'note': 'Used simplified save method'
            }
        except Exception as e:
            raise ValueError(f"Unable to save community results: {e}")
