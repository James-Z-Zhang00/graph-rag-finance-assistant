import time
from graphdatascience import GraphDataScience
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
        self.config = config or GDSConfig()
        self.gds = GraphDataScience(
            self.config.uri,
            auth=(self.config.username, self.config.password),
        )
        self.graph = connection_manager.get_connection()
        self.projection_name = "entities"
        self.G = None

        # Performance monitoring
        self.projection_time = 0
        self.knn_time = 0
        self.wcc_time = 0
        self.query_time = 0

        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes to optimize query performance."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.wcc)"
        ]
        connection_manager.create_multiple_indexes(index_queries)

    @timer
    def create_entity_projection(self) -> Tuple[Any, Dict[str, Any]]:
        """Create an in-memory GDS projection subgraph of entities."""
        start_time = time.time()
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"Error dropping existing projection (ignorable): {e}")
        entity_count = self._get_entity_count()
        if entity_count == 0:
            print("No valid entity nodes found.")
            return None, {"status": "error", "message": "No entities found"}
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name, "__Entity__", "*", nodeProperties=["embedding"]
            )
        except Exception as e:
            print(f"Error creating projection: {e}")
            try:
                print("Retrying projection with conservative configuration...")
                config = {
                    "nodeProjection": {"__Entity__": {"properties": ["embedding"]}},
                    "relationshipProjection": {"*": {"orientation": "UNDIRECTED"}},
                    "nodeProperties": ["embedding"]
                }
                self.G, result = self.gds.graph.project(self.projection_name, config)
            except Exception as e2:
                print(f"Second attempt also failed: {e2}")
                return None, {"status": "error", "message": str(e2)}
        self.projection_time = time.time() - start_time
        if self.G:
            print(f"Projection created successfully in {self.projection_time:.2f}s")
            return self.G, result
        return None, {"status": "error", "message": "Failed to create projection"}

    def _get_entity_count(self) -> int:
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
        """Detect similar entities using KNN algorithm (requires GDS)."""
        if not self.G:
            raise ValueError("Please create the entity projection first")
        start_time = time.time()
        print(f"Starting similar entity detection via GDS KNN (top_k={self.config.top_k}, threshold={self.config.similarity_threshold})...")
        try:
            top_k = max(1, self.config.top_k)
            mutate_result = self.gds.knn.mutate(
                self.G, nodeProperties=['embedding'], mutateRelationshipType='SIMILAR',
                mutateProperty='score', similarityCutoff=self.config.similarity_threshold, topK=top_k
            )
            write_result = self.gds.knn.write(
                self.G, nodeProperties=['embedding'], writeRelationshipType='SIMILAR',
                writeProperty='score', similarityCutoff=self.config.similarity_threshold, topK=top_k
            )
            self.knn_time = time.time() - start_time
            print(f"KNN complete: wrote {write_result['relationshipsWritten']} relationships in {self.knn_time:.2f}s")
            return {"status": "success", "relationshipsWritten": write_result['relationshipsWritten'], "knnTime": self.knn_time}
        except Exception as e:
            print(f"KNN algorithm failed: {e}")
            try:
                print("Retrying KNN with fallback parameters...")
                fallback_top_k = max(1, self.config.top_k // 2)
                fallback_result = self.gds.knn.write(
                    self.G, nodeProperties=["embedding"], writeRelationshipType="SIMILAR",
                    writeProperty="score", similarityCutoff=self.config.similarity_threshold,
                    topK=fallback_top_k, sampleRate=0.5
                )
                self.knn_time = time.time() - start_time
                return {"status": "success", "relationshipsWritten": fallback_result['relationshipsWritten'],
                        "knnTime": self.knn_time, "note": "Used fallback parameters"}
            except Exception as e2:
                return {"status": "error", "message": str(e)}

    @timer
    def detect_communities(self) -> Dict[str, Any]:
        """Detect communities using WCC algorithm (requires GDS)."""
        if not self.G:
            raise ValueError("Please create the entity projection first")
        start_time = time.time()
        print("Starting community detection via GDS WCC...")
        try:
            result = self.gds.wcc.write(
                self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"], consecutiveIds=True
            )
            self.wcc_time = time.time() - start_time
            community_count = result.get("communityCount", 0)
            print(f"Community detection complete: found {community_count} communities in {self.wcc_time:.2f}s")
            return {"status": "success", "communityCount": community_count, "wccTime": self.wcc_time}
        except Exception as e:
            print(f"WCC algorithm failed: {e}")
            try:
                print("Retrying WCC with fallback parameters...")
                fallback_result = self.gds.wcc.write(self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"])
                self.wcc_time = time.time() - start_time
                community_count = fallback_result.get("communityCount", 0)
                return {"status": "success", "communityCount": community_count,
                        "wccTime": self.wcc_time, "note": "Used fallback parameters"}
            except Exception as e2:
                return {"status": "error", "message": str(e)}

    @timer
    def find_potential_duplicates(self) -> List[Any]:
        """Find potentially duplicate entities."""
        query_start = time.time()

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

        results = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 1
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            WITH distinct
                [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id]
                AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
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
        """Execute the full entity processing pipeline."""
        start_time = time.time()
        duplicates = []

        try:
            # 1. Create entity projection
            self.G, projection_result = self.create_entity_projection()
            if not self.G:
                print("Entity projection creation failed — cannot continue")
                return [], {"status": "error", "message": "Projection creation failed"}

            # 2. Detect similar entities via GDS KNN
            knn_result = self.detect_similar_entities()
            if knn_result.get('status') == 'error':
                print(f"Similar entity detection failed: {knn_result.get('message')}")
                return [], {"status": "error", "message": "Similar entity detection failed"}

            # 3. Detect communities via GDS WCC
            wcc_result = self.detect_communities()
            if wcc_result.get('status') == 'error':
                print(f"Community detection failed: {wcc_result.get('message')}")
                return [], {"status": "error", "message": "Community detection failed"}

            # 4. Find potential duplicates
            duplicates = self.find_potential_duplicates()

            total_time = time.time() - start_time

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

        finally:
            self.cleanup()
