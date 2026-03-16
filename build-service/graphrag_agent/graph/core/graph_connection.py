from typing import Any, Optional
from graphrag_agent.config.neo4jdb import get_db_manager

class GraphConnectionManager:
    """
    Graph database connection manager.
    Responsible for creating and managing Neo4j connections, ensuring connection reuse.
    """

    _instance = None

    def __new__(cls):
        """Singleton implementation — ensures only one connection manager instance is created."""
        if cls._instance is None:
            cls._instance = super(GraphConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the connection manager; only executes on first creation."""
        if not getattr(self, "_initialized", False):
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            self._initialized = True

    def get_connection(self):
        """
        Get the graph database connection.

        Returns:
            Object connected to the Neo4j database
        """
        return self.graph

    def refresh_schema(self):
        """Refresh the graph database schema."""
        self.graph.refresh_schema()

    def execute_query(self, query: str, params: Optional[dict] = None) -> Any:
        """
        Execute a graph database query.

        Args:
            query: Query string
            params: Query parameters

        Returns:
            Query results
        """
        return self.graph.query(query, params or {})

    def create_index(self, index_query: str) -> None:
        """
        Create an index.

        Args:
            index_query: Index creation query
        """
        self.graph.query(index_query)

    def create_multiple_indexes(self, index_queries: list) -> None:
        """
        Create multiple indexes.

        Args:
            index_queries: List of index creation queries
        """
        for query in index_queries:
            self.create_index(query)

    def drop_index(self, index_name: str) -> None:
        """
        Drop an index.

        Args:
            index_name: Index name
        """
        try:
            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
            print(f"Dropped index {index_name} (if it existed)")
        except Exception as e:
            print(f"Error dropping index {index_name} (ignorable): {e}")

    def drop_all_indexes(self) -> None:
        """
        Drop all indexes (including regular and vector indexes).
        Call this before starting a build to ensure all old indexes are removed.
        """
        print("\n" + "="*60)
        print("Dropping all indexes...")
        print("="*60)

        try:
            # Get all indexes
            result = self.graph.query("""
                SHOW INDEXES
                YIELD name, type
                RETURN name, type
            """)

            if result:
                print(f"Found {len(result)} indexes, dropping...")

                for index_info in result:
                    index_name = index_info.get('name')
                    index_type = index_info.get('type', 'UNKNOWN')

                    if index_name:
                        try:
                            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
                            print(f"  Dropped index: {index_name} (type: {index_type})")
                        except Exception as e:
                            print(f"  Failed to drop index {index_name}: {e}")

                print(f"\nIndex cleanup complete, dropped {len(result)} indexes")
            else:
                print("No indexes found")

        except Exception as e:
            print(f"Error retrieving index list: {e}")
            print("Attempting to drop common index names...")

            # Fallback: try dropping common index names
            common_indexes = [
                "chunk_embedding",
                "chunk_vector",
                "entity_embedding",
                "entity_vector",
                "vector"
            ]

            for index_name in common_indexes:
                try:
                    self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
                    print(f"  Attempted to drop: {index_name}")
                except Exception as e:
                    print(f"  Failed to drop {index_name}: {e}")

        print("="*60 + "\n")

# Global connection manager instance
connection_manager = GraphConnectionManager()
