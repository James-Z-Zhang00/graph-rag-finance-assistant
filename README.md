# GraphRAG Finance Assistant

A production-grade, distributed RAG system for financial document intelligence. Answers natural language questions over SEC filings (10-K, 10-Q, 8-K) using a hybrid knowledge graph + vector search pipeline, with an enterprise compliance layer and RAGAS-based evaluation framework.

**[GitHub](https://github.com/James-Z-Zhang00/graph-rag-finance-assistant)**

---

## Why GraphRAG for Finance?

Standard dense retrieval flattens documents into chunks and misses cross-filing relationships — e.g. how a competitor's revenue trend affects a company's risk disclosures. GraphRAG builds a knowledge graph of entities, relationships, and thematic communities from the filings, enabling both precise local lookup (entity-level facts) and broad thematic reasoning (community-level context) in a single query.

---

## Architecture

```
  Streamlit Frontend
          |
    api-gateway :8000          ← single entry point; no RAG logic
     |              |
     |              └── Neo4j :7687  (graph CRUD, reasoning)
     |
  search-service :8003         ← LangGraph state machine: retrieve → generate → validate
     |              |
  Neo4j :7687   llm-gateway :8002  ← OpenAI-compatible proxy; key isolation
                    |
                 OpenAI API

  build-service :8004          ← async 6-stage graph build pipeline
     |              |
  sec-parser :8001  llm-gateway :8002
  (file parsing)    (entity extraction, embeddings)
     |
  Neo4j :7687
```

### Services

| Service | Port | Role |
|---------|------|------|
| `api-gateway` | 8000 | Frontend entry point; proxies chat, executes graph CRUD and reasoning |
| `sec-parser` | 8001 | Stateless file parser — PDF, HTML/XBRL, TXT → structured JSON |
| `llm-gateway` | 8002 | OpenAI-compatible proxy with key isolation, latency and token logging |
| `search-service` | 8003 | Hybrid GraphRAG retrieval + LangGraph answer pipeline |
| `build-service` | 8004 | Async graph construction orchestrator (6-stage pipeline) |
| `frontend` | — | Streamlit chat UI with knowledge graph visualizer |

---

## Key Engineering

### Hybrid GraphRAG Search Pipeline

The search-service runs a **LangGraph `StateGraph`** for each query:

```
START → agent → retrieve → generate → validate → END
```

- **`agent` node** — keyword extractor that decides which tools to call (local entity search vs. global community search)
- **`retrieve` node** — `HybridSearchTool` executes entity/relationship traversal, XBRL fact lookup, and Leiden community queries in Neo4j
- **`generate` node** — builds a grounded prompt from `RetrievalResult` objects with full citation provenance
- **`validate` node** — LLM-based faithfulness scorer (0.0–1.0); answers below 0.7 are regenerated with a grounding-enforced prompt

Local search resolves named entities and traverses their 1–2-hop graph neighborhood. Global search queries Leiden community summary nodes for thematic context not captured by entity lookup alone.

### SEC Filing Parser

A standalone microservice that parses raw filings into structured, graph-ready JSON through a 4-stage pipeline:

| Stage | Component | What it does |
|-------|-----------|-------------|
| 1 | `SecFileReader` | Routes PDF / HTML / XBRL / TXT to format-specific readers; extracts raw text, tables, XBRL facts |
| 2 | `FilingNormalizer` | Strips PDF artifacts, HTML tags, SEC boilerplate; normalizes whitespace |
| 3 | `SectionExtractor` | Regex-parses Item headings; isolates per-section text with character offsets |
| 4 | `TextChunker` | Tokenizes with `tiktoken` (cl100k_base); produces 500-token overlapping chunks tagged with `source_section` |

Each `TextChunk` carries its originating SEC Item number, enabling **section-aware retrieval** — the graph knows whether a chunk came from Item 1A (Risk Factors) vs. Item 7 (MD&A) and weights accordingly.

XBRL inline facts (`ix:nonFraction`) are extracted from HTML filings and stored as typed `FinancialFact` nodes (concept, value, period, unit, CIK) for precise numeric queries.

### Knowledge Graph Build Pipeline

`build-service` orchestrates a 6-stage async job:

1. **`gcs_download`** — optional GCS file pull
2. **`sec_parse`** — batch-sends files to sec-parser
3. **`drop_indexes`** — clean-slate Neo4j rebuild
4. **`build_graph`** — LLM entity and relationship extraction; writes `__Entity__`, `__Chunk__`, `__Document__`, `FinancialFact`, `Table`, `FilingSection` nodes
5. **`index_community`** — entity deduplication, Leiden community detection (via Neo4j GDS), LLM-generated community summaries
6. **`chunk_index`** — vector index build over `__Chunk__` nodes for semantic search

An **incremental mode** diffs the file manifest and re-processes only changed documents, avoiding a full graph rebuild.

### Enterprise Compliance Layer

Every query passes through three compliance components:

| Component | What it does |
|-----------|-------------|
| `PIIMasker` | Dual-layer: Microsoft Presidio for in-process NER-based redaction + Google Cloud DLP for output scanning; replaces detected tokens with `[PII:TYPE]` before any caching or LLM call |
| `AuditLogger` | Writes structured JSONL events per query lifecycle (`query_received`, `retrieval_done`, `generation_done`, `pii_masked`, `validation_done`) to `/tmp/audit_logs/audit_YYYY-MM-DD.jsonl` |
| `HallucinationValidator` | LLM faithfulness scorer; low-scoring answers are regenerated — score and `audit_id` are surfaced in every API response |

The PII layer was validated across **110 end-to-end test cases** covering nested PII detection inside existing placeholders, address formats with abbreviations, and edge cases across phone, SSN, credit card, and account number patterns.

### Multi-Layer Cache

| Layer | Backend | Description |
|-------|---------|-------------|
| L1 | In-memory dict | Process-local; fastest; session-scoped |
| L2 (local) | FAISS + JSON | Persisted across restarts; semantic similarity threshold 0.9 |
| L2 (cloud) | Upstash Redis | Distributed; TTL-managed (default 7 days) |
| L3 | Upstash Vector | Semantic deduplication for near-duplicate queries |

Set `CACHE_BACKEND=upstash` to enable L2/L3 cloud layers.

### LLM Gateway

A thin, stateless OpenAI-compatible proxy (port 8002) that:
- Validates an internal `GATEWAY_API_KEY` and injects the upstream credential — internal services never store the upstream key
- Logs model name, latency, prompt tokens, and completion tokens per request
- Accepts any OpenAI-compatible upstream (Azure, One-API, local Ollama) via `UPSTREAM_BASE_URL`

Substituting `gpt-4o` → `gpt-4.1-nano` for the search service reduced estimated output-token costs by **~96%** based on OpenAI published pricing, with faithfulness scores maintained within evaluation tolerance.

---

## Evaluation

The `eval/` directory contains a RAGAS-based evaluation framework with two experiments:

**Experiment A — GraphRAG vs. Naive Retrieval**
Runs the same finance QA dataset through the full hybrid GraphRAG pipeline and a dense-only baseline. Scores Faithfulness, Response Relevancy, Factual Correctness, and LLM Context Recall.

**Experiment B — Model Comparison**
Retrieves context via GraphRAG then generates answers with two models (e.g. `gpt-4o` vs. `gpt-4.1-nano`) on identical context, isolating generation quality from retrieval quality.

Results are saved as CSV and JSON to `eval/results/` with per-question breakdowns and aggregate delta tables.

---

## Tech Stack

| Area | Technologies |
|------|-------------|
| **Orchestration** | LangGraph (StateGraph), LangChain, LangChain-OpenAI, LangChain-Neo4j |
| **LLM / Embeddings** | OpenAI GPT-4o / GPT-4.1-nano, text-embedding-3-large |
| **Knowledge Graph** | Neo4j, Neo4j Graph Data Science (Leiden community detection) |
| **Vector Search** | FAISS (local), Upstash Vector (cloud) |
| **SEC Parsing** | pdfplumber, img2table, camelot, BeautifulSoup4, tiktoken |
| **Compliance** | Microsoft Presidio, Google Cloud DLP |
| **Evaluation** | RAGAS, pandas |
| **Cache** | In-memory, FAISS+JSON, Upstash Redis, Upstash Vector |
| **API** | FastAPI, Uvicorn, httpx |
| **Frontend** | Streamlit |
| **Infra** | Google Cloud Run, GCS, GitHub Actions CI/CD, Docker |

---

## Deployment

Each service is independently containerized and deployed to **Google Cloud Run** via GitHub Actions. Service-to-service authentication uses Google OIDC identity tokens (injected automatically in Cloud Run; disabled locally).

CI/CD workflows live in `.github/workflows/`:

```
deploy-api-gateway.yml
deploy-search-service.yml
deploy-build-service.yml
deploy-llm-gateway.yml
deploy-sec-parser.yml
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Neo4j 5.x (local or AuraDB)
- OpenAI API key

### Start services (in order)

```bash
# 1. LLM Gateway
cd llm-gateway && cp .env.example .env  # fill UPSTREAM_API_KEY
python main.py

# 2. SEC Parser
cd sec-parser && cp .env.example .env
python main.py

# 3. Build Service
cd build-service && cp .env.example .env  # fill NEO4J credentials
python main.py

# 4. Search Service
cd search-service && cp .env.example .env
python main.py

# 5. API Gateway
cd api-gateway && cp .env.example .env
python main.py

# 6. Frontend
cd frontend && streamlit run app.py
```

Each service exposes interactive API docs at `http://localhost:<port>/docs`.

### Trigger a graph build

```bash
# Upload SEC filing files
curl -X POST http://localhost:8004/files/upload \
  -F "files=@your_10k.html"

# Start a full build
curl -X POST http://localhost:8004/build/full

# Poll status
curl http://localhost:8004/build/jobs/<job_id>
```

### Run evaluation

```bash
cd eval
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY, NEO4J_*, SEARCH_SERVICE_URL

python run_eval.py --experiment graphrag_vs_naive
```

---

## Repository Structure

```
graph-rag-finance-assistant/
├── api-gateway/        # Entry point; graph CRUD and reasoning
├── build-service/      # Async graph construction pipeline
├── eval/               # RAGAS evaluation framework
├── frontend/           # Streamlit chat + KG visualizer
├── llm-gateway/        # OpenAI-compatible LLM proxy
├── search-service/     # LangGraph hybrid RAG pipeline + compliance
└── sec-parser/         # SEC filing parser microservice
```
