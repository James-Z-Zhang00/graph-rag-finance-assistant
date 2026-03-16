from typing import Dict, Any
import pandas as pd
from neo4j import GraphDatabase, Result
from langchain_neo4j import Neo4jGraph
from graphrag_agent.config.settings import NEO4J_CONFIG


class DBConnectionManager:
    """Neo4j database connection manager — singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.neo4j_uri = NEO4J_CONFIG["uri"]
        self.neo4j_username = NEO4J_CONFIG["username"]
        self.neo4j_password = NEO4J_CONFIG["password"]
        self.max_pool_size = NEO4J_CONFIG["max_pool_size"]

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password),
            max_connection_pool_size=self.max_pool_size
        )

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            refresh_schema=NEO4J_CONFIG["refresh_schema"],
        )

        self.session_pool = []
        self._initialized = True

    def get_driver(self):
        """Return the Neo4j driver instance."""
        return self.driver

    def get_graph(self):
        """Return the LangChain Neo4j graph instance."""
        return self.graph

    def execute_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """Execute a Cypher query and return results as a DataFrame."""
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )

    def get_session(self):
        """Get a session from the pool, or create a new one."""
        if self.session_pool:
            return self.session_pool.pop()
        else:
            return self.driver.session()

    def release_session(self, session):
        """Return a session to the pool, or close it if the pool is full."""
        if len(self.session_pool) < self.max_pool_size:
            self.session_pool.append(session)
        else:
            session.close()

    def close(self):
        """Close all sessions and the driver."""
        for session in self.session_pool:
            try:
                session.close()
            except:
                pass
        self.session_pool = []
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_db_manager: DBConnectionManager | None = None


def get_db_manager() -> DBConnectionManager:
    """Return the global database connection manager instance (lazy init)."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DBConnectionManager()
    return _db_manager
