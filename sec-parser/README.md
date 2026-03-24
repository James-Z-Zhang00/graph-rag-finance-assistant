# sec-parser

SEC filing parser for 10-K, 10-Q, and 8-K filings. Runs on port **8001**.

## Overview

The sec-parser is a standalone microservice that receives SEC filing files (PDF, HTML/XBRL, TXT) and produces structured, semantically-enriched JSON suitable for downstream graph construction. It extracts narrative text, document sections by Item number, financial tables, and XBRL-tagged numeric facts. It has no database connections — it parses files and returns data.

---

## Architecture Position

```
   build-service :8004
         |
         | POST /api/sec/parse (multipart files)
         ↓
   sec-parser :8001
         |
         └── returns structured JSON
             (text, chunks, tables, XBRL facts, sections, metadata)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sec/parse` | Parse uploaded files; returns structured JSON |
| `POST` | `/api/sec/process` | Parse files from a directory path; streams results to `INGEST_URL` |
| `GET` | `/api/healthz` | Health check |

### `POST /api/sec/parse`

Accepts `multipart/form-data` file uploads. Returns parsed documents as JSON.

```
Request:  multipart/form-data
          files:            one or more SEC filing files
          pdf_table_method: "img2table" (default) | "camelot"

Response: {
  "documents": [
    {
      "filepath": "string",
      "filename": "string",
      "form_type": "10-K | 10-Q | 8-K | UNKNOWN",
      "cik": "string",
      "company_name": "string",
      "filing_date": "string",
      "content": "string",              // full normalized text
      "content_length": int,
      "chunks": [["token", ...]],       // tokenized text chunks
      "chunk_count": int,
      "numeric_facts": [{...}],         // XBRL-tagged numeric values
      "tables": [{...}],                // extracted tables with cells
      "sections": [                     // document sections by Item number
        {"item": "1", "title": "...", "content": "..."}
      ]
    }
  ]
}
```

---

## Supported File Formats

| Format | What Gets Extracted |
|--------|---------------------|
| **PDF** | Text (page-by-page, avoids table regions), tables via `img2table` or `camelot` |
| **HTML / HTM** | Text, tables with full header resolution, XBRL-enriched cell values |
| **XBRL** | Treated as HTML; extracts all `ix:nonFraction` Inline XBRL semantic facts |
| **TXT / MD** | Plain text only |

---

## 4-Stage Processing Pipeline

Each file passes through these stages in sequence:

| Stage | Component | What it Does |
|-------|-----------|-------------|
| 1 | `SecFileReader` | Detects file type, routes to format-specific reader, populates raw content, tables, and XBRL facts |
| 2 | `FilingNormalizer` | Strips PDF artifacts, HTML tags, SEC boilerplate; normalises whitespace |
| 3 | `SectionExtractor` | Regex-parses Item headings, detects form type, extracts per-section text |
| 4 | `TextChunker` | Tokenises with tiktoken (cl100k_base), creates overlapping 500-token chunks |

---

## Key Models

```python
class FileType(Enum):
    PDF, HTML, HTM, XBRL, TXT, MD

class FormType(Enum):
    FORM_10K, FORM_10Q, FORM_8K, UNKNOWN

class XBRLNumeric(BaseModel):
    name: str           # XBRL concept, e.g. "us-gaap:RevenueFromContractWithCustomer"
    value: float        # computed: raw × 10^scale with sign applied
    raw: str
    context_ref: str
    entity: Optional[str]       # CIK
    period_start: Optional[str]
    period_end: Optional[str]
    unit: Optional[str]         # "USD", "shares", etc.
    scale: int
    fact_id: Optional[str]

class TableCell(BaseModel):
    value: str
    row: int
    col: int
    column_header: Optional[str]
    row_header: Optional[str]
    xbrl: Optional[XBRLNumeric]  # present for HTML/XBRL tables only

class ExtractedTable(BaseModel):
    table_id: str
    cells: List[TableCell]
    caption: Optional[str]
    section: Optional[str]
    source: str           # "pdf" or "html"
    page: Optional[int]

class FilingSection(BaseModel):
    item: str             # "1", "1A", "7", "2.01", etc.
    title: str
    content: str
    start: int            # character offset in normalized text
    end: int
```

---

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SEC_PARSER_HOST` | `0.0.0.0` | No | Bind address |
| `SEC_PARSER_PORT` | `8001` | No | Listen port |
| `SEC_PARSER_RELOAD` | `false` | No | Hot reload (dev only) |
| `SEC_PARSER_LOG_LEVEL` | `info` | No | Log verbosity |
| `CHUNK_SIZE` | `500` | No | Tokens per text chunk |
| `CHUNK_OVERLAP` | `100` | No | Overlap tokens between chunks |
| `MAX_TEXT_LENGTH` | `500000` | No | Max chars before pre-split |
| `INGEST_URL` | `http://localhost:8000/api/ingest/sec` | No | Downstream URL for `/process` endpoint |

---

## Local Development

```bash
cd sec-parser

pip install -r requirements.txt

python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive API docs: `http://localhost:8001/docs`

---

## Docker

Build and run from the **project root**:

```bash
docker build -f sec-parser/Dockerfile.sec-parser -t sec-parser .

docker run -p 8001:8001 sec-parser
```

The image installs `libgl1`, `libglib2.0-0`, and `ghostscript` for PDF and image-based table extraction.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | Web framework and ASGI server |
| `pdfplumber` | PDF text extraction |
| `img2table` | Image-based PDF table detection (default) |
| `camelot-py` | Lattice/stream PDF table extraction (alternative) |
| `opencv-contrib-python-headless` | Image processing for table detection |
| `beautifulsoup4` | HTML/XBRL DOM parsing |
| `tiktoken` | GPT-compatible tokenisation for chunking |
| `python-multipart` | Multipart file upload parsing |
| `requests` | HTTP POST to downstream ingest service |
