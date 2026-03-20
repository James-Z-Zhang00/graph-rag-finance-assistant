# [GDS] This entire module is commented out for Aura Free compatibility.
# [GDS] GraphProjectionMixin wraps gds.graph.project() calls which require GDS,
# [GDS] unavailable on AuraDB Free. Re-enable when switching back to GDS.

# [GDS] from typing import Dict, Any, Tuple
# [GDS]
# [GDS] class GraphProjectionMixin:
# [GDS]     """Mixin class for graph projection functionality."""
# [GDS]
# [GDS]     def create_projection(self) -> Tuple[Any, Dict]:
# [GDS]         """Create graph projection."""
# [GDS]         print("Creating graph projection for community detection...")
# [GDS]
# [GDS]         node_count = self._get_node_count()
# [GDS]         if node_count > self.node_count_limit:
# [GDS]             print(f"Warning: node count ({node_count}) exceeds limit ({self.node_count_limit})")
# [GDS]             return self._create_filtered_projection(node_count)
# [GDS]
# [GDS]         try:
# [GDS]             self.gds.graph.drop(self.projection_name, failIfMissing=False)
# [GDS]         except Exception as e:
# [GDS]             print(f"Error dropping old projection (ignorable): {e}")
# [GDS]
# [GDS]         try:
# [GDS]             self.G, result = self.gds.graph.project(
# [GDS]                 self.projection_name,
# [GDS]                 "__Entity__",
# [GDS]                 {
# [GDS]                     "_ALL_": {
# [GDS]                         "type": "*",
# [GDS]                         "orientation": "UNDIRECTED",
# [GDS]                         "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
# [GDS]                     }
# [GDS]                 },
# [GDS]             )
# [GDS]             print(f"Graph projection created: {result.get('nodeCount', 0)} nodes, "
# [GDS]                   f"{result.get('relationshipCount', 0)} relationships")
# [GDS]             return self.G, result
# [GDS]         except Exception as e:
# [GDS]             print(f"Standard projection failed: {e}")
# [GDS]             return self._create_conservative_projection()
# [GDS]
# [GDS]     def _get_node_count(self) -> int:
# [GDS]         result = self.graph.query(
# [GDS]             "MATCH (e:__Entity__) RETURN count(e) AS count"
# [GDS]         )
# [GDS]         return result[0]["count"] if result else 0
# [GDS]
# [GDS]     def _create_filtered_projection(self, total_node_count: int) -> Tuple[Any, Dict]:
# [GDS]         print("Creating filtered projection...")
# [GDS]         try:
# [GDS]             result = self.graph.query(
# [GDS]                 """
# [GDS]                 MATCH (e:__Entity__)-[r]-()
# [GDS]                 WITH e, count(r) AS rel_count
# [GDS]                 ORDER BY rel_count DESC
# [GDS]                 LIMIT toInteger($limit)
# [GDS]                 RETURN collect(id(e)) AS important_nodes
# [GDS]                 """,
# [GDS]                 params={"limit": self.node_count_limit}
# [GDS]             )
# [GDS]             if not result or not result[0]["important_nodes"]:
# [GDS]                 return self._create_conservative_projection()
# [GDS]             important_nodes = result[0]["important_nodes"]
# [GDS]             config = {
# [GDS]                 "nodeProjection": {
# [GDS]                     "__Entity__": {
# [GDS]                         "properties": ["*"],
# [GDS]                         "filter": f"id(node) IN {important_nodes}"
# [GDS]                     }
# [GDS]                 },
# [GDS]                 "relationshipProjection": {
# [GDS]                     "_ALL_": {
# [GDS]                         "type": "*",
# [GDS]                         "orientation": "UNDIRECTED",
# [GDS]                         "properties": {"weight": {"property": "*", "aggregation": "COUNT"}}
# [GDS]                     }
# [GDS]                 }
# [GDS]             }
# [GDS]             self.G, result = self.gds.graph.project(self.projection_name, config)
# [GDS]             print(f"Filtered projection created: {result.get('nodeCount', 0)} nodes, "
# [GDS]                   f"{result.get('relationshipCount', 0)} relationships")
# [GDS]             return self.G, result
# [GDS]         except Exception as e:
# [GDS]             print(f"Filtered projection failed: {e}")
# [GDS]             return self._create_conservative_projection()
# [GDS]
# [GDS]     def _create_conservative_projection(self) -> Tuple[Any, Dict]:
# [GDS]         print("Trying to create projection with conservative configuration...")
# [GDS]         try:
# [GDS]             config = {
# [GDS]                 "nodeProjection": "__Entity__",
# [GDS]                 "relationshipProjection": "*"
# [GDS]             }
# [GDS]             self.G, result = self.gds.graph.project(self.projection_name, config)
# [GDS]             print(f"Conservative projection created: {result.get('nodeCount', 0)} nodes")
# [GDS]             return self.G, result
# [GDS]         except Exception as e:
# [GDS]             print(f"Conservative projection failed: {e}")
# [GDS]             return self._create_minimal_projection()
# [GDS]
# [GDS]     def _create_minimal_projection(self) -> Tuple[Any, Dict]:
# [GDS]         print("Trying to create minimal projection...")
# [GDS]         try:
# [GDS]             result = self.graph.query(
# [GDS]                 """
# [GDS]                 MATCH (e:__Entity__)-[r]-()
# [GDS]                 WITH e, count(r) AS rel_count
# [GDS]                 ORDER BY rel_count DESC
# [GDS]                 LIMIT 1000
# [GDS]                 RETURN collect(id(e)) AS critical_nodes
# [GDS]                 """
# [GDS]             )
# [GDS]             if not result or not result[0]["critical_nodes"]:
# [GDS]                 raise ValueError("Unable to retrieve critical nodes")
# [GDS]             critical_nodes = result[0]["critical_nodes"]
# [GDS]             minimal_config = {
# [GDS]                 "nodeProjection": {
# [GDS]                     "__Entity__": {
# [GDS]                         "filter": f"id(node) IN {critical_nodes}"
# [GDS]                     }
# [GDS]                 },
# [GDS]                 "relationshipProjection": "*"
# [GDS]             }
# [GDS]             self.G, result = self.gds.graph.project(self.projection_name, minimal_config)
# [GDS]             print(f"Minimal projection created: {result.get('nodeCount', 0)} nodes")
# [GDS]             return self.G, result
# [GDS]         except Exception as e:
# [GDS]             print(f"All projection methods failed: {e}")
# [GDS]             raise ValueError("Unable to create the required graph projection")

# Stub so existing imports don't break
from typing import Dict, Any, Tuple

class GraphProjectionMixin:
    """Stub — all GDS projection methods commented out for Aura Free compatibility."""
    pass
