from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time

from langchain_core.tools import BaseTool

from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.cache_manager.manager import CacheManager, ContextAndKeywordAwareCacheKeyStrategy, MemoryCacheBackend
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.search.utils import VectorUtils
from graphrag_agent.config.settings import BASE_SEARCH_CONFIG


class BaseSearchTool(ABC):
    """Base class for search tools, providing common functionality for all search implementations."""

    def __init__(self, cache_dir: str = "./cache/search"):
        """
        Initialize the search tool.

        Args:
            cache_dir: Cache directory for storing search results
        """
        # Initialize the LLM and embeddings model
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        self.default_vector_limit = BASE_SEARCH_CONFIG["vector_limit"]
        self.default_text_limit = BASE_SEARCH_CONFIG["text_limit"]
        self.default_semantic_top_k = BASE_SEARCH_CONFIG["semantic_top_k"]
        self.default_relevance_top_k = BASE_SEARCH_CONFIG["relevance_top_k"]

        # Initialize cache manager
        self.cache_manager = CacheManager(
            key_strategy=ContextAndKeywordAwareCacheKeyStrategy(),
            storage_backend=MemoryCacheBackend(
                max_size=BASE_SEARCH_CONFIG["cache_max_size"]
            ),
            cache_dir=cache_dir
        )

        # Performance monitoring metrics
        self.performance_metrics = {
            "query_time": 0,  # Database query time
            "llm_time": 0,    # LLM processing time
            "total_time": 0   # Total processing time
        }

        # Initialize Neo4j connection
        self._setup_neo4j()

    def _setup_neo4j(self):
        """Set up the Neo4j connection."""
        # Get the database connection manager
        db_manager = get_db_manager()

        # Get the graph database instance
        self.graph = db_manager.get_graph()

        # Get the driver (for direct query execution)
        self.driver = db_manager.get_driver()

    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """
        Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            Query results
        """
        # Execute query via the connection manager
        return get_db_manager().execute_query(cypher, params)

    @abstractmethod
    def _setup_chains(self):
        """
        Set up processing chains; must be implemented by subclasses.
        Used to configure LLM processing chains and prompt templates.
        """
        pass

    @abstractmethod
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Extract keywords from a query.

        Args:
            query: Query string

        Returns:
            Dict[str, List[str]]: Keyword dictionary containing low-level and high-level keywords
        """
        pass

    @abstractmethod
    def search(self, query: Any) -> str:
        """
        Execute a search.

        Args:
            query: Query content; may be a string or a dict containing additional information

        Returns:
            str: Search results
        """
        pass

    def vector_search(self, query: str, limit: int = None) -> List[str]:
        """
        Vector similarity-based search method.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List[str]: List of matching entity IDs
        """
        try:
            limit = limit or self.default_vector_limit
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Build Neo4j vector search query
            cypher = """
            CALL db.index.vector.queryNodes('vector', $limit, $embedding)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score DESC
            """

            # Execute search
            results = self.db_query(cypher, {
                "embedding": query_embedding,
                "limit": limit
            })

            # Extract entity IDs
            if not results.empty:
                return results['id'].tolist()
            else:
                return []

        except Exception as e:
            print(f"Vector search failed: {e}")
            # If vector search fails, fall back to text search
            return self.text_search(query, limit)

    def text_search(self, query: str, limit: int = None) -> List[str]:
        """
        Text matching-based search method (fallback for vector search).

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List[str]: List of matching entity IDs
        """
        try:
            limit = limit or self.default_text_limit
            # Build full-text search query
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """

            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })

            if not results.empty:
                return results['id'].tolist()
            else:
                return []

        except Exception as e:
            print(f"Text search failed: {e}")
            return []

    def semantic_search(self, query: str, entities: List[Dict],
                        embedding_field: str = "embedding",
                        top_k: int = None) -> List[Dict]:
        """
        Perform semantic similarity search over a set of entities.

        Args:
            query: Search query
            entities: List of entities; each item must contain the field specified by embedding_field
            embedding_field: Field name holding the embedding vector
            top_k: Maximum number of results to return

        Returns:
            List of entities sorted by similarity, each augmented with a "score" field
        """
        try:
            top_k = top_k or self.default_semantic_top_k
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Rank using VectorUtils
            return VectorUtils.rank_by_similarity(
                query_embedding,
                entities,
                embedding_field,
                top_k
            )
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return entities[:top_k] if top_k else entities

    def filter_by_relevance(self, query: str, docs: List, top_k: int = None) -> List:
        """
        Filter documents by relevance to the query.

        Args:
            query: Query string
            docs: List of documents
            top_k: Maximum number of results to return

        Returns:
            List of documents sorted by relevance
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            limit = top_k or self.default_relevance_top_k
            return VectorUtils.filter_documents_by_relevance(
                query_embedding,
                docs,
                top_k=limit
            )
        except Exception as e:
            print(f"Document filtering failed: {e}")
            limit = top_k or self.default_relevance_top_k
            return docs[:limit] if limit else docs

    def get_tool(self) -> BaseTool:
        """
        Get the search tool instance.

        Returns:
            BaseTool: Search tool
        """
        # Create a dynamic tool class
        class DynamicSearchTool(BaseTool):
            name: str = f"{self.__class__.__name__.lower()}"
            description: str = "Advanced search tool for retrieving information from the knowledge base"

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("Async execution not implemented")

        return DynamicSearchTool()

    def _log_performance(self, operation: str, start_time: float):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            start_time: Start time
        """
        duration = time.time() - start_time
        self.performance_metrics[operation] = duration
        print(f"Performance metric - {operation}: {duration:.4f}s")

    def close(self):
        """Close resource connections."""
        # Close the Neo4j connection
        if hasattr(self, 'graph'):
            # Call close() if Neo4jGraph supports it
            if hasattr(self.graph, 'close'):
                self.graph.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — ensures resources are properly released."""
        self.close()
