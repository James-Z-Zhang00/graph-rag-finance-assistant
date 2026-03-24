# build-service

Knowledge graph construction pipeline for SEC filings. Runs on port **8004**.

## Overview

The build-service orchestrates the end-to-end pipeline that transforms raw SEC filing files into a queryable Neo4j knowledge graph. It manages asynchronous build jobs across six stages: file download (optional GCS), SEC parsing, graph construction (entity/relationship extraction via LLM), community detection (Leiden algorithm), and vector chunk indexing. An incremental mode updates only changed documents without a full rebuild.

---

## Architecture Position

```
      Files (local upload or GCS bucket)
              |
        build-service :8004
              |
    ┌─────────┴──────────────┐
    |                        |
sec-parser :8001       llm-gateway :8002
(parse filings)        (entity extraction LLM,
                        embeddings)
    |                        |
    └─────────┬──────────────┘
              |
           Neo4j :7687
      (knowledge graph store)
```

---

## Endpoints

### Build Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/build/full` | Start a full 6-stage graph build; returns `job_id` immediately |
| `POST` | `/build/incremental` | Start an incremental update (changed files only) |
| `GET` | `/build/jobs` | List all jobs (history) |
| `GET` | `/build/jobs/{job_id}` | Poll job status and progress |
| `GET` | `/build/check` | Check connectivity to Neo4j, sec-parser, and llm-gateway |

### File Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/files/upload` | Upload one or more SEC filing files |
| `GET` | `/files` | List uploaded files with metadata |
| `DELETE` | `/files/{filename}` | Delete an uploaded file |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |

---

## Build Pipeline Stages

A full build (`POST /build/full`) runs these stages in order:

| # | Stage Name | Description |
|---|-----------|-------------|
| 1 | `gcs_download` | *(Optional)* Download files from GCS bucket to temp directory |
| 2 | `sec_parse` | Send files to sec-parser; receive structured JSON (text, tables, XBRL, sections) |
| 3 | `drop_indexes` | Drop all existing Neo4j indexes (clean slate) |
| 4 | `build_graph` | LLM-based entity and relationship extraction; write nodes and edges to Neo4j |
| 5 | `index_community` | Create entity indexes, detect and merge duplicate entities, run community detection (Leiden), generate community summaries via LLM |
| 6 | `chunk_index` | Build vector indexes on Chunk nodes for semantic search |

An **incremental build** runs `detect_changes` → `incremental_update`, updating only documents that changed since the last run.

---

## Job Status Model

Poll `GET /build/jobs/{job_id}` to track progress.

```python
class JobResponse(BaseModel):
    job_id: str
    job_type: str                  # "full" | "incremental"
    status: str                    # "pending" | "running" | "completed" | "failed"
    stage: str                     # current stage name (e.g. "build_graph", "done")
    started_at: Optional[float]    # Unix timestamp
    completed_at: Optional[float]
    error: Optional[str]           # full traceback on failure
    stats: Optional[Dict]          # e.g. documents_parsed, entities_integrated, total_time
```

---

## Neo4j Schema Written

**Nodes**

| Label | Description |
|-------|-------------|
| `__Entity__` | Named entities (Company, Executive, Filing, Product, Geography, etc.) |
| `__Chunk__` | Text passages with vector embeddings for semantic search |
| `__Community__` | Entity clusters from community detection, with LLM-generated summaries |
| `__Document__` | Source filing metadata |
| `FinancialFact` | XBRL-tagged numeric values (linked from `__Document__`) |
| `Table` | Extracted financial tables (linked from `__Document__`) |
| `FilingSection` | SEC form sections by Item number (linked from `__Document__`) |

**Relationships**: `REPORTED_IN`, `HAS_REVENUE`, `OPERATES_IN`, `RISK_AFFECTS`, `MANAGES`, `OWNS`, `COMPETES_WITH`, `FILED_BY`, `EMPLOYED_BY`, `SEGMENT_OF`, `PERIOD_COVERS`, `HAS_FACT`, `HAS_TABLE`, `HAS_SECTION`, `MENTIONS`

---

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `BUILD_SERVICE_PORT` | `8004` | No | Listen port |
| `SEC_PARSER_URL` | `http://localhost:8001` | Yes | sec-parser endpoint |
| `OPENAI_BASE_URL` | `http://localhost:8002/v1` | Yes | llm-gateway endpoint |
| `OPENAI_API_KEY` | — | Yes | Gateway API key |
| `OPENAI_LLM_MODEL` | `gpt-4o` | No | LLM model for entity extraction |
| `OPENAI_EMBEDDINGS_MODEL` | `text-embedding-3-large` | No | Embeddings model |
| `NEO4J_URI` | `neo4j://localhost:7687` | Yes | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | `12345678` | Yes | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | No | Database name |
| `GCS_BUCKET_NAME` | *(empty)* | No | GCS bucket; leave empty to use local files |
| `GCS_FILES_PREFIX` | `small/` | No | Object prefix within GCS bucket |
| `FILES_DIR` | `./files` | No | Local directory for uploaded files |
| `GRAPH_COMMUNITY_ALGORITHM` | `leiden` | No | `leiden` or `sllpa` |
| `BATCH_SIZE` | `100` | No | General processing batch size |
| `EMBEDDING_BATCH_SIZE` | `64` | No | Embeddings batch size |
| `MAX_WORKERS` | `4` | No | Parallel worker threads |

**.env.example** (included in repo — copy and fill in):

```env
BUILD_SERVICE_PORT=8004
SEC_PARSER_URL=http://localhost:8001
OPENAI_BASE_URL=http://localhost:8002/v1
OPENAI_API_KEY=your-gateway-key
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

---

## Local Development

```bash
cd build-service

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your credentials

python main.py
```

Interactive API docs: `http://localhost:8004/docs`

Use `GET /build/check` to verify all downstream services are reachable before triggering a build.

---

## Docker

Build and run from the **project root**:

```bash
docker build -f build-service/Dockerfile.build-service -t build-service .

docker run -p 8004:8004 --env-file build-service/.env build-service
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | Web framework and ASGI server |
| `langchain` / `langchain_openai` | LLM orchestration for entity extraction |
| `langchain_neo4j` | Neo4j query integration |
| `langgraph` | Agent workflow (used in graph construction agents) |
| `neo4j` | Neo4j Python driver |
| `graphdatascience` | Community detection and entity similarity (Leiden, GDS) |
| `sentence_transformers` | Text embeddings for chunk indexing |
| `faiss-cpu` | Local vector similarity search |
| `google-cloud-storage` | GCS file download (optional) |
| `rich` | Progress bars and terminal output |
