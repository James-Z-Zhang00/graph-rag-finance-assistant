from typing import Dict, Any, Tuple

class GraphProjectionMixin:
    """Mixin class for graph projection functionality."""

    def create_projection(self) -> Tuple[Any, Dict]:
        """Create graph projection."""
        print("Creating graph projection for community detection...")

        # Check node count
        node_count = self._get_node_count()
        if node_count > self.node_count_limit:
            print(f"Warning: node count ({node_count}) exceeds limit ({self.node_count_limit})")
            return self._create_filtered_projection(node_count)

        # Drop any existing projection
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"Error dropping old projection (ignorable): {e}")

        # Create standard projection
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name,
                "__Entity__",
                {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                    }
                },
            )
            print(f"Graph projection created: {result.get('nodeCount', 0)} nodes, "
                  f"{result.get('relationshipCount', 0)} relationships")
            return self.G, result
        except Exception as e:
            print(f"Standard projection failed: {e}")
            return self._create_conservative_projection()

    def _get_node_count(self) -> int:
        """Get node count."""
        result = self.graph.query(
            "MATCH (e:__Entity__) RETURN count(e) AS count"
        )
        return result[0]["count"] if result else 0

    def _create_filtered_projection(self, total_node_count: int) -> Tuple[Any, Dict]:
        """Create a filtered projection."""
        print("Creating filtered projection...")

        try:
            # Get high-degree nodes
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT toInteger($limit)
                RETURN collect(id(e)) AS important_nodes
                """,
                params={"limit": self.node_count_limit}
            )

            if not result or not result[0]["important_nodes"]:
                return self._create_conservative_projection()

            important_nodes = result[0]["important_nodes"]

            # Create filtered projection
            config = {
                "nodeProjection": {
                    "__Entity__": {
                        "properties": ["*"],
                        "filter": f"id(node) IN {important_nodes}"
                    }
                },
                "relationshipProjection": {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}}
                    }
                }
            }

            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"Filtered projection created: {result.get('nodeCount', 0)} nodes, "
                  f"{result.get('relationshipCount', 0)} relationships")
            return self.G, result

        except Exception as e:
            print(f"Filtered projection failed: {e}")
            return self._create_conservative_projection()

    def _create_conservative_projection(self) -> Tuple[Any, Dict]:
        """Create a projection with conservative configuration."""
        print("Trying to create projection with conservative configuration...")

        try:
            # Use minimal configuration
            config = {
                "nodeProjection": "__Entity__",
                "relationshipProjection": "*"
            }

            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"Conservative projection created: {result.get('nodeCount', 0)} nodes")
            return self.G, result

        except Exception as e:
            print(f"Conservative projection failed: {e}")
            return self._create_minimal_projection()

    def _create_minimal_projection(self) -> Tuple[Any, Dict]:
        """Create a minimal projection."""
        print("Trying to create minimal projection...")

        try:
            # Get most important nodes
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT 1000
                RETURN collect(id(e)) AS critical_nodes
                """
            )

            if not result or not result[0]["critical_nodes"]:
                raise ValueError("Unable to retrieve critical nodes")

            critical_nodes = result[0]["critical_nodes"]

            # Create minimal projection
            minimal_config = {
                "nodeProjection": {
                    "__Entity__": {
                        "filter": f"id(node) IN {critical_nodes}"
                    }
                },
                "relationshipProjection": "*"
            }

            self.G, result = self.gds.graph.project(
                self.projection_name,
                minimal_config
            )
            print(f"Minimal projection created: {result.get('nodeCount', 0)} nodes")
            return self.G, result

        except Exception as e:
            print(f"All projection methods failed: {e}")
            raise ValueError("Unable to create the required graph projection")
