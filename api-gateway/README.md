# api-gateway

Single entry point for the GraphRAG Finance Assistant frontend. Runs on port **8000**.

## Overview

The api-gateway is a FastAPI service that acts as the sole interface between the frontend and the rest of the system. It does two things: proxies chat queries to the **search-service** (which owns all RAG and LLM logic), and executes knowledge graph operations directly against **Neo4j** (entity/relation CRUD, graph reasoning, source retrieval).

It intentionally contains no LLM or retrieval logic. All intelligence lives downstream.

## Architecture Position

```
Frontend (Streamlit)
        |
  api-gateway :8000          ← this service
   |
   ├── /api/chat ──────────────────→ search-service :8003
   |                                  (RAG pipeline, LLM, hybrid search)
   |
   └── /api/knowledge_graph ───────→ Neo4j :7687
       /api/entity/*                  (graph queries, CRUD, reasoning)
       /api/relation/*
       /api/source
```

---

## Endpoints

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send a question; returns full answer with citations and quality score |
| `POST` | `/api/chat/stream` | Same as above, streamed as plain text chunks |
| `POST` | `/api/clear` | Clear session state for a given `session_id` |

### Knowledge Graph

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/knowledge_graph` | Fetch graph nodes and links (query params: `limit`, `query`) |
| `GET` | `/api/knowledge_graph_from_message` | Extract KG subgraph mentioned in a text message |
| `GET` | `/api/chunks` | Paginated document chunks (params: `limit`, `offset`) |
| `GET` | `/api/entity_types` | List all entity type labels in the graph |
| `GET` | `/api/relation_types` | List all relationship type labels |
| `POST` | `/api/kg_reasoning` | Execute a named graph reasoning algorithm |

**Reasoning types** (passed as `reasoning_type` in `ReasoningRequest`):

| Type | Description | Required fields |
|------|-------------|-----------------|
| `entity_community` | Community detection around an entity | `entity_a` |
| `shortest_path` | Shortest path between two entities | `entity_a`, `entity_b` |
| `one_two_hop` | 1–2 hop paths between two entities | `entity_a`, `entity_b` |
| `common_neighbors` | Shared neighbors of two entities | `entity_a`, `entity_b` |
| `all_paths` | All paths up to `max_depth` (default 3) | `entity_a`, `entity_b` |
| `entity_cycles` | Cycles involving an entity | `entity_a` |
| `entity_influence` | Influence propagation from an entity | `entity_a` |

### Entity CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/entities/search` | Search entities by term or type |
| `POST` | `/api/entity/create` | Create a new entity node |
| `POST` | `/api/entity/update` | Update entity properties |
| `POST` | `/api/entity/delete` | Delete an entity node |

### Relation CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/relations/search` | Search relationships by source, target, or type |
| `POST` | `/api/relation/create` | Create a relationship between two entities |
| `POST` | `/api/relation/update` | Update a relationship |
| `POST` | `/api/relation/delete` | Delete a relationship |

### Source Content

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/source` | Fetch text content for a source ID |
| `POST` | `/api/source_info` | Fetch file metadata for a source ID |
| `POST` | `/api/content_batch` | Batch fetch content for multiple chunk IDs |
| `POST` | `/api/source_info_batch` | Batch fetch metadata for multiple source IDs |

### Other

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/feedback` | Record positive/negative feedback for an answer |
| `GET` | `/api/health` | Health check |

---

## Key Models

```python
# Chat request
class ChatRequest(BaseModel):
    message: str
    session_id: str
    debug: bool = False           # includes execution_log in response when True
    agent_type: str = "naive_rag_agent"
    use_deeper_tool: bool = True
    show_thinking: bool = False

# Chat response
class ChatResponse(BaseModel):
    answer: str
    execution_log: Optional[List[Dict]]   # returned only when debug=True
    kg_data: Optional[Dict]               # graph nodes/links extracted from answer
    citations: Optional[List[str]]        # provenance: source IDs from retrieval
    quality_score: Optional[float]        # faithfulness score 0.0–1.0 from validator
    audit_id: Optional[str]               # links to audit trail entry

# Graph reasoning request
class ReasoningRequest(BaseModel):
    reasoning_type: str
    entity_a: str
    entity_b: Optional[str] = None
    max_depth: Optional[int] = 3
    algorithm: Optional[str] = "leiden"
```

---

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `GATEWAY_HOST` | `0.0.0.0` | No | Bind address |
| `GATEWAY_PORT` | `8000` | No | Listen port |
| `GATEWAY_RELOAD` | `false` | No | Hot reload (dev only) |
| `GATEWAY_LOG_LEVEL` | `info` | No | Log verbosity |
| `SEARCH_SERVICE_URL` | `http://localhost:8003` | Yes | Upstream search-service URL |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Yes | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | — | Yes | Neo4j password |
| `NEO4J_MAX_POOL_SIZE` | `10` | No | Connection pool size |
| `GRAPH_COMMUNITY_ALGORITHM` | `leiden` | No | Community detection algorithm (`leiden` or `louvain`) |

**.env.example**

```env
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
SEARCH_SERVICE_URL=http://localhost:8003
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

---

## Local Development

```bash
cd api-gateway

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Neo4j credentials and service URLs

# Start the service
python main.py
# or with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs available at `http://localhost:8000/docs` once running.

---

## Docker

Build and run from the **project root**:

```bash
docker build -f api-gateway/Dockerfile.api-gateway -t api-gateway .

docker run -p 8000:8000 --env-file api-gateway/.env api-gateway
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework and routing |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation |
| `httpx` | Async HTTP client for proxying to search-service |
| `neo4j` | Neo4j driver for graph queries |
| `pandas` | DataFrame transformation of Cypher query results |
| `google-auth` | OIDC token generation for Cloud Run auth |
| `python-dotenv` | `.env` file loading |

---

## Service-to-Service Authentication

Locally, no authentication is required between services.

On **Google Cloud Run**, requests from api-gateway to search-service are automatically authenticated using Google OIDC identity tokens. The gateway detects the Cloud Run environment via the `K_SERVICE` environment variable and injects an `Authorization: Bearer <token>` header on all upstream requests.
