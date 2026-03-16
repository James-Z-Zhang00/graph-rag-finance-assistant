from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from graphrag_agent.config.settings import GDS_CONCURRENCY

class SLLPADetector(GraphProjectionMixin, BaseCommunityDetector):
    """Community detection implementation using the SLLPA algorithm."""

    def detect_communities(self) -> Dict[str, Any]:
        """Run SLLPA algorithm community detection."""
        if not self.G:
            raise ValueError("Please create the graph projection first")

        print("Starting SLLPA community detection...")

        try:
            # Execute SLLPA algorithm
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                **self._get_optimized_sllpa_params()
            )

            community_count = result.get('communityCount', 0)
            iterations = result.get('iterations', 0)

            print(f"SLLPA complete: {community_count} communities, "
                  f"{iterations} iterations")

            return {
                'communityCount': community_count,
                'iterations': iterations
            }

        except Exception as e:
            print(f"SLLPA algorithm failed: {e}")
            return self._execute_fallback_sllpa()

    def _execute_fallback_sllpa(self) -> Dict[str, Any]:
        """Execute fallback SLLPA algorithm."""
        print("Trying with fallback parameters...")

        try:
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                maxIterations=50,        # Reduce iteration count
                minAssociationStrength=0.2,  # Increase threshold
                concurrency=1            # Single-thread execution
            )

            return {
                'communityCount': result.get('communityCount', 0),
                'iterations': result.get('iterations', 0),
                'note': 'Used fallback parameters'
            }
        except Exception as e:
            raise ValueError(f"SLLPA algorithm failed: {e}")

    def _get_optimized_sllpa_params(self) -> Dict[str, Any]:
        """Get optimized SLLPA parameters."""
        if self.memory_mb > 32 * 1024:  # >32GB
            return {
                'maxIterations': 100,
                'minAssociationStrength': 0.05,
                'concurrency': GDS_CONCURRENCY
            }
        elif self.memory_mb > 16 * 1024:  # >16GB
            return {
                'maxIterations': 80,
                'minAssociationStrength': 0.08,
                'concurrency': max(1, GDS_CONCURRENCY - 1)
            }
        else:  # Low memory
            return {
                'maxIterations': 50,
                'minAssociationStrength': 0.1,
                'concurrency': max(1, GDS_CONCURRENCY // 2)
            }

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
