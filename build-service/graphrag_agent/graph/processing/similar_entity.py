import time
# [GDS] from graphdatascience import GraphDataScience  # GDS - commented out for Aura Free compatibility
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass

from graphrag_agent.config.settings import (
    similarity_threshold,
    BATCH_SIZE,
    GDS_MEMORY_LIMIT,
    NEO4J_CONFIG,
    SIMILAR_ENTITY_SETTINGS,
)
from graphrag_agent.graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

@dataclass
class GDSConfig:
    """Neo4j GDS configuration parameters."""
    uri: str = NEO4J_CONFIG["uri"]
    username: str = NEO4J_CONFIG["username"]
    password: str = NEO4J_CONFIG["password"]
    similarity_threshold: float = similarity_threshold
    word_edit_distance: int = SIMILAR_ENTITY_SETTINGS["word_edit_distance"]
    batch_size: int = SIMILAR_ENTITY_SETTINGS["batch_size"]
    memory_limit: int = SIMILAR_ENTITY_SETTINGS["memory_limit"]  # Unit: GB
    top_k: int = SIMILAR_ENTITY_SETTINGS["top_k"]

    def __post_init__(self):
        # Use config values if provided
        if BATCH_SIZE:
            self.batch_size = BATCH_SIZE
        if GDS_MEMORY_LIMIT:
            self.memory_limit = GDS_MEMORY_LIMIT

class SimilarEntityDetector:
    """
    Similar entity detector using the Neo4j GDS library for entity similarity
    analysis and community detection.

    Main capabilities:
    1. Build an in-memory entity projection graph
    2. Use the KNN algorithm to identify similar entities
    3. Use the WCC algorithm for community detection
    4. Identify potentially duplicate entities
    """

    def __init__(self, config: GDSConfig = None):
        """
        Initialize the similar entity detector.

        Args:
            config: GDS configuration parameters including connection info and algorithm thresholds
        """
        self.config = config or GDSConfig()
        # [GDS] self.gds = GraphDataScience(
        # [GDS]     self.config.uri,
        # [GDS]     auth=(self.config.username, self.config.password),
        # [GDS]     aura_ds=True
        # [GDS] )
        self.graph = connection_manager.get_connection()
        # [GDS] self.projection_name = "entities"
        # [GDS] self.G = None

        # Performance monitoring
        self.projection_time = 0
        self.knn_time = 0
        self.wcc_time = 0
        self.query_time = 0

        # Create indexes to optimize duplicate entity detection
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes to optimize query performance."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.wcc)"
        ]

        connection_manager.create_multiple_indexes(index_queries)

    # [GDS] @timer
    # [GDS] def create_entity_projection(self) -> Tuple[Any, Dict[str, Any]]:
    # [GDS]     """Create an in-memory GDS projection subgraph of entities (requires GDS)."""
    # [GDS]     start_time = time.time()
    # [GDS]     try:
    # [GDS]         self.gds.graph.drop(self.projection_name, failIfMissing=False)
    # [GDS]     except Exception as e:
    # [GDS]         print(f"Error dropping existing projection (ignorable): {e}")
    # [GDS]     entity_count = self._get_entity_count()
    # [GDS]     if entity_count == 0:
    # [GDS]         print("No valid entity nodes found.")
    # [GDS]         return None, {"status": "error", "message": "No entities found"}
    # [GDS]     try:
    # [GDS]         self.G, result = self.gds.graph.project(
    # [GDS]             self.projection_name, "__Entity__", "*", nodeProperties=["embedding"]
    # [GDS]         )
    # [GDS]     except Exception as e:
    # [GDS]         print(f"Error creating projection: {e}")
    # [GDS]         try:
    # [GDS]             print("Retrying projection with conservative configuration...")
    # [GDS]             config = {
    # [GDS]                 "nodeProjection": {"__Entity__": {"properties": ["embedding"]}},
    # [GDS]                 "relationshipProjection": {"*": {"orientation": "UNDIRECTED"}},
    # [GDS]                 "nodeProperties": ["embedding"]
    # [GDS]             }
    # [GDS]             self.G, result = self.gds.graph.project(self.projection_name, config)
    # [GDS]         except Exception as e2:
    # [GDS]             print(f"Second attempt also failed: {e2}")
    # [GDS]             return None, {"status": "error", "message": str(e2)}
    # [GDS]     self.projection_time = time.time() - start_time
    # [GDS]     if self.G:
    # [GDS]         print(f"Projection created successfully in {self.projection_time:.2f}s")
    # [GDS]         return self.G, result
    # [GDS]     return None, {"status": "error", "message": "Failed to create projection"}

    def _get_entity_count(self) -> int:
        """
        Get the total number of entities.

        Returns:
            int: Entity count
        """
        result = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL
            RETURN count(e) AS count
            """
        )
        return result[0]["count"] if result else 0

    @timer
    def detect_similar_entities(self) -> Dict[str, Any]:
        """Detect similar entities using the vector index (pure Cypher, Aura-compatible).
        Replaces gds.knn which requires GDS unavailable on AuraDB Free.
        Uses db.index.vector.queryNodes() on the existing entity embedding vector index.
        """
        start_time = time.time()
        top_k = max(1, self.config.top_k)
        print(f"Starting similar entity detection via vector index (top_k={top_k}, threshold={self.config.similarity_threshold})...")

        try:
            result = self.graph.query(
                """
                MATCH (e:`__Entity__`)
                WHERE e.embedding IS NOT NULL
                CALL db.index.vector.queryNodes('vector', $topK, e.embedding)
                YIELD node AS similar, score
                WHERE similar <> e AND score >= $threshold
                MERGE (e)-[r:SIMILAR]->(similar)
                ON CREATE SET r.score = score
                ON MATCH SET r.score = CASE WHEN r.score < score THEN score ELSE r.score END
                RETURN count(*) AS relationshipsWritten
                """,
                params={
                    'topK': top_k,
                    'threshold': self.config.similarity_threshold
                }
            )
            self.knn_time = time.time() - start_time
            relationships_written = result[0]['relationshipsWritten'] if result else 0
            print(f"Vector similarity complete: {relationships_written} SIMILAR relationships in {self.knn_time:.2f}s")
            return {
                "status": "success",
                "relationshipsWritten": relationships_written,
                "knnTime": self.knn_time
            }

        except Exception as e:
            print(f"Vector similarity detection failed: {e}")
            return {"status": "error", "message": str(e)}

    # [GDS] @timer
    # [GDS] def detect_similar_entities_gds(self) -> Dict[str, Any]:
    # [GDS]     """Detect similar entities using KNN algorithm (requires GDS)."""
    # [GDS]     if not self.G:
    # [GDS]         raise ValueError("Please create the entity projection first")
    # [GDS]     start_time = time.time()
    # [GDS]     print("Starting similar entity detection...")
    # [GDS]     try:
    # [GDS]         top_k = max(1, self.config.top_k)
    # [GDS]         mutate_result = self.gds.knn.mutate(
    # [GDS]             self.G, nodeProperties=['embedding'], mutateRelationshipType='SIMILAR',
    # [GDS]             mutateProperty='score', similarityCutoff=self.config.similarity_threshold, topK=top_k
    # [GDS]         )
    # [GDS]         write_result = self.gds.knn.write(
    # [GDS]             self.G, nodeProperties=['embedding'], writeRelationshipType='SIMILAR',
    # [GDS]             writeProperty='score', similarityCutoff=self.config.similarity_threshold, topK=top_k
    # [GDS]         )
    # [GDS]         self.knn_time = time.time() - start_time
    # [GDS]         print(f"KNN complete: wrote {write_result['relationshipsWritten']} relationships in {self.knn_time:.2f}s")
    # [GDS]         return {"status": "success", "relationshipsWritten": write_result['relationshipsWritten'], "knnTime": self.knn_time}
    # [GDS]     except Exception as e:
    # [GDS]         print(f"KNN algorithm failed: {e}")
    # [GDS]         try:
    # [GDS]             print("Retrying KNN with fallback parameters...")
    # [GDS]             fallback_top_k = max(1, self.config.top_k // 2)
    # [GDS]             fallback_result = self.gds.knn.write(
    # [GDS]                 self.G, nodeProperties=["embedding"], writeRelationshipType="SIMILAR",
    # [GDS]                 writeProperty="score", similarityCutoff=self.config.similarity_threshold,
    # [GDS]                 topK=fallback_top_k, sampleRate=0.5
    # [GDS]             )
    # [GDS]             self.knn_time = time.time() - start_time
    # [GDS]             return {"status": "success", "relationshipsWritten": fallback_result['relationshipsWritten'],
    # [GDS]                     "knnTime": self.knn_time, "note": "Used fallback parameters"}
    # [GDS]         except Exception as e2:
    # [GDS]             return {"status": "error", "message": str(e)}

    @timer
    def detect_communities(self) -> Dict[str, Any]:
        """Detect communities using iterative Cypher WCC (pure Cypher, Aura-compatible).
        Replaces gds.wcc.write() which requires GDS unavailable on AuraDB Free.
        Propagates minimum component ID through SIMILAR relationships until convergence.
        """
        start_time = time.time()
        print("Starting community detection via iterative Cypher WCC...")

        # Initialize: each entity gets its own component ID
        self.graph.query("""
            MATCH (e:`__Entity__`)
            SET e.wcc = id(e)
        """)

        # Propagate minimum component ID through SIMILAR relationships until stable
        max_iterations = 20
        for i in range(max_iterations):
            result = self.graph.query("""
                MATCH (e:`__Entity__`)-[:SIMILAR]-(n:`__Entity__`)
                WITH e, min(n.wcc) AS min_wcc
                WHERE min_wcc < e.wcc
                SET e.wcc = min_wcc
                RETURN count(*) AS updates
            """)
            updates = result[0]['updates'] if result else 0
            if updates == 0:
                print(f"  WCC converged after {i + 1} iterations")
                break

        self.wcc_time = time.time() - start_time

        result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL
            RETURN count(DISTINCT e.wcc) AS communityCount
        """)
        community_count = result[0]['communityCount'] if result else 0
        print(f"WCC complete: {community_count} communities in {self.wcc_time:.2f}s")

        return {
            "status": "success",
            "communityCount": community_count,
            "wccTime": self.wcc_time
        }

    # [GDS] @timer
    # [GDS] def detect_communities_gds(self) -> Dict[str, Any]:
    # [GDS]     """Detect communities using WCC algorithm (requires GDS)."""
    # [GDS]     if not self.G:
    # [GDS]         raise ValueError("Please create the entity projection first")
    # [GDS]     start_time = time.time()
    # [GDS]     print("Starting community detection...")
    # [GDS]     try:
    # [GDS]         result = self.gds.wcc.write(
    # [GDS]             self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"], consecutiveIds=True
    # [GDS]         )
    # [GDS]         self.wcc_time = time.time() - start_time
    # [GDS]         community_count = result.get("communityCount", 0)
    # [GDS]         print(f"Community detection complete: found {community_count} communities in {self.wcc_time:.2f}s")
    # [GDS]         return {"status": "success", "communityCount": community_count, "wccTime": self.wcc_time}
    # [GDS]     except Exception as e:
    # [GDS]         print(f"WCC algorithm failed: {e}")
    # [GDS]         try:
    # [GDS]             print("Retrying WCC with fallback parameters...")
    # [GDS]             fallback_result = self.gds.wcc.write(self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"])
    # [GDS]             self.wcc_time = time.time() - start_time
    # [GDS]             community_count = fallback_result.get("communityCount", 0)
    # [GDS]             return {"status": "success", "communityCount": community_count,
    # [GDS]                     "wccTime": self.wcc_time, "note": "Used fallback parameters"}
    # [GDS]         except Exception as e2:
    # [GDS]             return {"status": "error", "message": str(e)}

    @timer
    def find_potential_duplicates(self) -> List[Any]:
        """
        Find potentially duplicate entities.

        Returns:
            List[Any]: List of candidate duplicate entity groups
        """
        query_start = time.time()

        # Find communities containing multiple entities
        community_counts = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL AND size(e.id) > 1
            WITH e.wcc AS community, count(*) AS count
            WHERE count > 1
            RETURN community, count
            ORDER BY count DESC
            """
        )

        if not community_counts:
            print("No communities with potential duplicate entities found")
            return []

        # Find potential duplicates in valid communities
        results = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 1  // length greater than 1 character
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance calculation
            WITH distinct
                [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id]
                AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // Merge groups that share common elements
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result,
                index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // Additional filtering
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult,
                combinedResultIndex,
                allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={'distance': self.config.word_edit_distance}
        )

        self.query_time = time.time() - query_start

        # Convert query results to a simple list-of-string-lists format
        processed_results = []
        for record in results:
            if "combinedResult" in record and isinstance(record["combinedResult"], list):
                processed_results.append(record["combinedResult"])

        print(f"Potential duplicate search complete: found {len(processed_results)} candidate groups in {self.query_time:.2f}s")

        return processed_results

    def cleanup(self) -> None:
        """Clean up the in-memory projection graph."""
        if self.G:
            try:
                self.G.drop()
                print("Projection graph cleaned up")
            except Exception as e:
                print(f"Error cleaning up projection graph: {str(e)}")
            finally:
                self.G = None

    @timer
    def process_entities(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Execute the full entity processing pipeline.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: List of potential duplicate entities and performance stats
        """
        start_time = time.time()
        duplicates = []

        try:
            # [GDS] 1. Create entity projection (not needed for pure Cypher)
            # [GDS] self.G, projection_result = self.create_entity_projection()
            # [GDS] if not self.G:
            # [GDS]     print("Entity projection creation failed — cannot continue")
            # [GDS]     return [], {"status": "error", "message": "Projection creation failed"}

            # 1. Detect similar entities via vector index
            knn_result = self.detect_similar_entities()

            if knn_result.get('status') == 'error':
                print(f"Similar entity detection failed: {knn_result.get('message')}")
                return [], {"status": "error", "message": "Similar entity detection failed"}

            # 3. Detect communities
            wcc_result = self.detect_communities()

            if wcc_result.get('status') == 'error':
                print(f"Community detection failed: {wcc_result.get('message')}")
                return [], {"status": "error", "message": "Community detection failed"}

            # 4. Find potential duplicates
            duplicates = self.find_potential_duplicates()

            total_time = time.time() - start_time

            # Prepare performance statistics
            time_records = {
                "Projection time": self.projection_time,
                "KNN time": self.knn_time,
                "WCC time": self.wcc_time,
                "Query time": self.query_time
            }

            stats = get_performance_stats(total_time, time_records)
            stats.update({
                "status": "success",
                "Candidate entity groups": len(duplicates),
                "Relationships written": knn_result.get('relationshipsWritten', 0),
                "Communities found": wcc_result.get('communityCount', 0)
            })

            print_performance_stats(stats)

            return duplicates, stats

        except Exception as e:
            print(f"Error during entity processing: {e}")
            return [], {"status": "error", "message": str(e)}

        # [GDS] finally:
        # [GDS]     # Always clean up the in-memory GDS projection graph
        # [GDS]     self.cleanup()
