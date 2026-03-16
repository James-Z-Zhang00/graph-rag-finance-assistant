"""
Standalone Neo4j connection manager for the API gateway.
Does not depend on graphrag_agent.
"""

from typing import Dict, Any
import pandas as pd
from neo4j import GraphDatabase, Result
from config.settings import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    NEO4J_MAX_POOL_SIZE, NEO4J_REFRESH_SCHEMA,
)


class DBConnectionManager:
    """Neo4j database connection manager — singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            max_connection_pool_size=NEO4J_MAX_POOL_SIZE,
        )
        self._initialized = True

    def get_driver(self):
        return self.driver

    def execute_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df,
        )

    def close(self):
        if self.driver:
            self.driver.close()


_db_manager = DBConnectionManager()


def get_db_manager() -> DBConnectionManager:
    return _db_manager
