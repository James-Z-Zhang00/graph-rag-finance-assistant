# search-service

GraphRAG hybrid search engine with enterprise compliance layer. Runs on port **8003**.

## Overview

The search-service hosts the core RAG pipeline for the GraphRAG Finance Assistant. It receives natural language questions, executes a hybrid retrieval strategy over the Neo4j knowledge graph (local entity/relationship traversal + global Leiden community search), generates grounded answers via LLM, and returns them with citation provenance and a faithfulness quality score. All requests pass through a PII masking and audit trail layer.

It does not serve the frontend directly — the api-gateway proxies all chat requests here.

---

## Architecture Position

```
  api-gateway :8000
        |
        | POST /search/hybrid
        ↓
  search-service :8003
        |
    ┌───┴──────────────────┐
    |                      |
 Neo4j :7687          llm-gateway :8002
 (graph retrieval)    (answer generation,
                       keyword extraction,
                       faithfulness scoring)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search/hybrid` | Execute hybrid search; returns full answer with compliance fields |
| `POST` | `/search/hybrid/stream` | Same as above, streamed as plain text chunks |
| `POST` | `/search/clear` | Clear in-memory cache for a session |
| `GET` | `/health` | Health check |

### Request / Response

```python
class SearchRequest(BaseModel):
    query: str
    session_id: str = "default"
    debug: bool = False       # includes execution_log when True

class SearchResponse(BaseModel):
    answer: str
    execution_log: Optional[List[Dict]]   # node-by-node trace (debug only)
    citations: Optional[List[str]]        # source provenance from retrieval
    quality_score: Optional[float]        # faithfulness score 0.0–1.0
    audit_id: Optional[str]               # links to JSONL audit trail entry
```

---

## Hybrid Search Architecture

Each request runs through a **LangGraph state machine**:

```
START → agent → retrieve → generate → validate → END
                  ↑
          (tools: HybridSearchTool,
                  GlobalSearchTool)
```

| Node | What it Does |
|------|-------------|
| `agent` | Extracts low-level and high-level keywords; decides which tools to call |
| `retrieve` | Calls `HybridSearchTool`: entity/relationship graph traversal + community lookup + XBRL financial facts + filing sections |
| `generate` | Invokes LLM with retrieved context; collects citations from `RetrievalResult` objects |
| `validate` | Scores answer faithfulness [0.0–1.0]; regenerates with grounding-enforced prompt if score < 0.7 |

**Local search** resolves keyword-matched entities in Neo4j, fetches their relationships and chunk text.
**Global search** queries Leiden community nodes for high-level thematic context.

---

## Compliance Layer

Every request passes through three compliance components before and after the RAG pipeline:

| Component | What it Does |
|-----------|-------------|
| `PIIMasker` | Regex-detects email, phone, SSN, credit card in query text; replaces with `[PII:TYPE]` before caching and LLM calls |
| `AuditLogger` | Writes structured JSONL events to `/tmp/audit_logs/audit_YYYY-MM-DD.jsonl` — `query_received`, `retrieval_done`, `generation_done`, `pii_masked`, `validation_done` |
| `HallucinationValidator` | LLM-based faithfulness scorer; answers below threshold are regenerated with a grounding-enforced prompt |

---

## Cache Architecture

| Layer | Mode | Backend | Description |
|-------|------|---------|-------------|
| L1 | Both | In-memory | Process-local dict, fastest, limited size |
| L2 | Local | Disk (FAISS + JSON) | Persisted across restarts, session-scoped |
| L2 | Upstash | Upstash Redis | Distributed, TTL-managed (default: 7 days) |
| L3 | Upstash | Upstash Vector | Semantic similarity matching (threshold: 0.9) |

Set `CACHE_BACKEND=upstash` and provide Upstash credentials to enable cloud caching.

---

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SEARCH_SERVICE_PORT` | `8003` | No | Listen port |
| `OPENAI_BASE_URL` | `http://localhost:8002/v1` | Yes | llm-gateway endpoint |
| `OPENAI_API_KEY` | — | Yes | Gateway API key |
| `OPENAI_LLM_MODEL` | `gpt-4.1-nano` | No | LLM model for search and generation |
| `OPENAI_EMBEDDINGS_MODEL` | `text-embedding-3-large` | No | Embeddings model |
| `NEO4J_URI` | — | Yes | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | — | Yes | Neo4j password |
| `CACHE_BACKEND` | `local` | No | `local` or `upstash` |
| `UPSTASH_REDIS_REST_URL` | — | If Upstash | Upstash Redis URL |
| `UPSTASH_REDIS_REST_TOKEN` | — | If Upstash | Upstash Redis token |
| `UPSTASH_VECTOR_REST_URL` | — | If Upstash | Upstash Vector URL |
| `UPSTASH_VECTOR_REST_TOKEN` | — | If Upstash | Upstash Vector token |
| `UPSTASH_CACHE_TTL` | `604800` | No | Cache TTL in seconds (7 days) |
| `TEMPERATURE` | `0.1` | No | LLM temperature |
| `MAX_TOKENS` | `10000` | No | Max tokens per LLM response |

**.env.example** (included in repo):

```env
SEARCH_SERVICE_PORT=8003
OPENAI_BASE_URL=http://localhost:8002/v1
OPENAI_API_KEY=your-gateway-key
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
CACHE_BACKEND=local
```

---

## Local Development

```bash
cd search-service

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your credentials

python main.py
```

Interactive API docs: `http://localhost:8003/docs`

---

## Docker

Build and run from the **project root**:

```bash
docker build -f search-service/Dockerfile.search-service -t search-service .

docker run -p 8003:8003 --env-file search-service/.env search-service
```

The Dockerfile pre-downloads the `all-MiniLM-L6-v2` sentence-transformer model at build time into `/app/cache/model`, avoiding HuggingFace network calls at runtime (important for Cloud Run cold starts).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | Web framework and ASGI server |
| `langgraph` | Agent state machine (retrieve → generate → validate workflow) |
| `langchain` / `langchain_openai` | LLM chains, prompt templates, tool binding |
| `langchain_neo4j` | Neo4j Cypher query integration |
| `neo4j` | Neo4j Python driver |
| `sentence_transformers` | Local embedding model for L1/L2 cache similarity |
| `faiss-cpu` | Local vector index for L2 cache |
| `upstash-redis` / `upstash-vector` | Optional cloud cache backends (L2/L3) |
| `google-auth` | OIDC token injection for Cloud Run service-to-service auth |
