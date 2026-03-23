# graph-rag-finance-assistant

#### Overview

Phase 4 delivered distributed microservices. Phase 5 adds an enterprise compliance layer:

- **Full citation provenance** — surface existing `RetrievalResult` citations through to API responses
- **Automated audit trails** — structured JSONL event logging per query lifecycle
- **PII masking** — regex-based detection and masking before caching/logging
- **Hallucination reduction** — LangGraph validation node scores faithfulness and regenerates low-quality answers