from fastapi import APIRouter
from neo4j import GraphDatabase
from graphrag_agent.config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

router = APIRouter()


@router.get("/health")
async def health():
    neo4j_ok = False
    neo4j_error = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        neo4j_ok = True
    except Exception as e:
        neo4j_error = str(e)

    status = "ok" if neo4j_ok else "degraded"
    return {
        "status": status,
        "service": "search-service",
        "neo4j": {"connected": neo4j_ok, "uri": NEO4J_URI, "error": neo4j_error},
    }
