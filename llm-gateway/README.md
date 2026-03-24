# llm-gateway

OpenAI-compatible LLM proxy with authentication and request logging. Runs on port **8002**.

## Overview

The llm-gateway is a thin, stateless proxy that sits between all internal services and the upstream LLM provider. It validates a gateway API key, swaps in the upstream provider credentials, forwards the request transparently, and logs model name, latency, and token counts. All internal services point to the gateway rather than calling OpenAI directly — giving a single place to rotate keys, switch providers, or add rate limiting.

---

## Architecture Position

```
  search-service :8003      build-service :8004
         |                          |
         | Bearer <GATEWAY_API_KEY> |
         └──────────┬───────────────┘
                    ↓
           llm-gateway :8002
                    |
                    | Bearer <UPSTREAM_API_KEY>
                    ↓
         OpenAI (or any OpenAI-compatible endpoint)
         e.g. api.openai.com, One-API, Azure OpenAI
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion — streaming and non-streaming |
| `POST` | `/v1/embeddings` | Embeddings generation |
| `GET` | `/health` | Health check; returns upstream base URL |

The `/v1/` prefix is intentional — all internal services configure `OPENAI_BASE_URL=http://llm-gateway:8002/v1` and use the standard OpenAI SDK without modification.

---

## Request / Response Models

Fully OpenAI-compatible. Any field accepted by OpenAI is passed through.

```python
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List] = None
    tool_choice: Optional[Any] = None
    # ... all other OpenAI fields supported via extra="allow"

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
```

---

## Authentication

Clients send a **Bearer token** in the `Authorization` header:

```
Authorization: Bearer <GATEWAY_API_KEY>
```

The gateway validates this against the `GATEWAY_API_KEY` environment variable and returns `401 Unauthorized` on mismatch. It then strips the gateway key and injects the `UPSTREAM_API_KEY` before forwarding to the provider — internal services never see or store the upstream credential.

If `GATEWAY_API_KEY` is left empty, authentication is disabled (useful for local development).

---

## Request Logging

Every request produces structured log lines:

```
# On request
INFO  chat request model=gpt-4.1-nano stream=False messages=4

# On non-streaming response (includes token counts)
INFO  chat done model=gpt-4.1-nano elapsed=1.23s prompt_tokens=842 completion_tokens=156

# On streaming response
INFO  chat stream done model=gpt-4.1-nano elapsed=2.47s

# On error
ERROR chat error model=gpt-4.1-nano: <upstream error>
```

Token counts are extracted from `response["usage"]` and are available for cost accounting in downstream metrics.

---

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `LLM_GATEWAY_PORT` | `8002` | No | Listen port |
| `LLM_GATEWAY_HOST` | `0.0.0.0` | No | Bind address |
| `LLM_GATEWAY_RELOAD` | `false` | No | Hot reload (dev only) |
| `LLM_GATEWAY_LOG_LEVEL` | `info` | No | Log verbosity |
| `GATEWAY_API_KEY` | *(empty)* | No | Key clients must send; empty = auth disabled |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Yes | Upstream LLM provider base URL |
| `UPSTREAM_API_KEY` | — | Yes | Upstream provider API key |
| `REQUEST_TIMEOUT` | `120` | No | Request timeout in seconds |

**.env.example** (included in repo):

```env
LLM_GATEWAY_PORT=8002
GATEWAY_API_KEY=your-internal-gateway-key
UPSTREAM_BASE_URL=https://api.openai.com/v1
UPSTREAM_API_KEY=sk-...
```

To use an OpenAI-compatible alternative (e.g. One-API, Azure, local Ollama):

```env
UPSTREAM_BASE_URL=https://your-one-api-endpoint/v1
UPSTREAM_API_KEY=your-one-api-key
```

---

## Local Development

```bash
cd llm-gateway

pip install -r requirements.txt

cp .env.example .env
# Edit .env with upstream credentials

python main.py
```

Interactive API docs: `http://localhost:8002/docs`

---

## Docker

Build and run from the **project root**:

```bash
docker build -f llm-gateway/Dockerfile.llm-gateway -t llm-gateway .

docker run -p 8002:8002 --env-file llm-gateway/.env llm-gateway
```

The image is intentionally minimal — no ML models, no databases. It is stateless and cold-starts instantly.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | Web framework and ASGI server |
| `httpx` | Async HTTP client for upstream forwarding (streaming + non-streaming) |
| `pydantic` | Request/response validation |
| `python-dotenv` | `.env` file loading |
