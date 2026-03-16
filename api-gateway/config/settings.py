"""
API Gateway configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
GATEWAY_RELOAD = os.getenv("GATEWAY_RELOAD", "false").lower() == "true"
GATEWAY_LOG_LEVEL = os.getenv("GATEWAY_LOG_LEVEL", "info")

# Upstream search service
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8003")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_MAX_POOL_SIZE = int(os.getenv("NEO4J_MAX_POOL_SIZE", "10"))
NEO4J_REFRESH_SCHEMA = os.getenv("NEO4J_REFRESH_SCHEMA", "false").lower() == "true"

COMMUNITY_ALGORITHM = os.getenv("GRAPH_COMMUNITY_ALGORITHM", "leiden")
