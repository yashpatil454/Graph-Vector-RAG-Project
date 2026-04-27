# Graph + Vector RAG API

A hybrid **Retrieval-Augmented Generation (RAG)** system that combines **FAISS vector similarity search** with a **Neo4j knowledge graph** to produce rich, structured context for LLM consumption. Documents are ingested from PDFs, chunked, embedded with Google Gemini, and simultaneously used to extract semantic triples that populate a graph database.

---

## Architecture Overview

```
PDFs
 │
 ▼
┌─────────────────────┐
│   Data Processor    │  Loads, cleans & chunks PDFs via LangChain
│  (PyPDF + tiktoken) │  Persists chunks as JSONL to data/processed_chunks/
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌─────────┐  ┌──────────────────┐
│  FAISS  │  │  Knowledge Graph │
│ Vector  │  │ (Neo4j + Gemini  │
│  Store  │  │   2.5 Flash LLM) │
│(Gemini  │  │  Extracts SPO    │
│Embeddings│  │  triples → MERGE │
└────┬────┘  └────────┬─────────┘
     │                │
     └───────┬────────┘
             ▼
    ┌─────────────────┐
    │  Hybrid Fusion  │  Combines vector hits + KG traversal
    │    Service      │  into a single LLM-ready context string
    └─────────────────┘
```

---

## Features

- **PDF Ingestion** — Load single or entire directories of PDFs; parallel processing via `ProcessPoolExecutor`
- **Token-aware chunking** — Uses `tiktoken` (`text-embedding-3-small`) for accurate token-boundary splits
- **Gemini Embeddings** — `models/gemini-embedding-001` with disk-based `CacheBackedEmbeddings` to avoid redundant API calls
- **FAISS Vector Store** — Persisted locally; async-safe operations run off the event loop via `asyncio.to_thread`
- **Knowledge Graph (Neo4j)** — Gemini 2.5 Flash extracts `(subject, predicate, object)` triples; MERGE semantics prevent duplicates
- **Hybrid Fusion** — Single `/hybrid_fusion/fuse` endpoint returns vector hits, graph relationships, and a combined context block
- **Structured Logging** — Singleton `RotatingFileHandler` logger + console output; per-request middleware captures latency
- **FastAPI** — Full OpenAPI docs auto-generated at `/rag/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| PDF parsing | PyPDF (via LangChain) |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` + tiktoken |
| Embeddings | Google Gemini (`langchain-google-genai`) |
| Vector search | FAISS (`faiss-cpu`) |
| LLM (triple extraction) | Gemini 2.5 Flash (`langchain-google-genai`) |
| Graph database | Neo4j (async driver) |
| Config | `pydantic-settings` + `.env` |
| Caching | LangChain `CacheBackedEmbeddings` + `LocalFileStore` |

---

## Project Structure

```
├── app/
│   ├── main.py                        # FastAPI app entry point
│   ├── core/
│   │   ├── config.py                  # Settings loaded from .env via pydantic-settings
│   │   ├── logger.py                  # Singleton rotating-file + console logger
│   │   └── logging_middleware.py      # Per-request latency logging middleware
│   ├── models/
│   │   └── request_models.py          # Pydantic request/response models + dataclasses
│   ├── routers/
│   │   ├── health.py                  # GET /health
│   │   ├── data_processor_router.py   # POST /data_processor/process
│   │   ├── vector_store_router.py     # GET  /load_vector_store/initialize_load_vector_store
│   │   ├── knowledge_graph_router.py  # POST /knowledge_graph/build, GET /knowledge_graph/query
│   │   └── hybrid_fusion_router.py    # POST /hybrid_fusion/fuse
│   └── services/
│       ├── data_processor.py          # PDFProcessor + singleton factory
│       ├── vector_store_service.py    # Async FAISS VectorStoreService + singleton factory
│       ├── knowledge_graph_service.py # Async Neo4j KnowledgeGraphService + singleton factory
│       └── hybrid_fusion_service.py   # HybridFusionService combining vector + graph
├── data/
│   ├── processed_chunks/              # JSONL chunks persisted after PDF processing
│   └── processed_triples/             # JSONL triples streamed during KG build
├── vector_store/
│   └── cache/                         # Disk cache for Gemini embeddings
├── logs/
│   └── app.log                        # Rotating log file
├── tests/                             # Demo scripts and unit tests
├── docs/                              # Additional guides
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- A running **Neo4j** instance (local or cloud) — default `bolt://localhost:7687`
- A **Google AI Studio** API key with access to Gemini models

---

## Setup

### 1. Clone and create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Optional overrides
APP_ROOT_PATH=/rag
PDF_DATA_DIR=data
PDF_CHUNK_SIZE=1000
PDF_CHUNK_OVERLAP=200
EMBEDDING_MODEL=models/gemini-embedding-001
VECTOR_STORE_PATH=vector_store
VECTOR_SEARCH_K=4
LOG_LEVEL=INFO
```

### 4. Place PDF files

Copy your PDF documents into the `data/` directory (or a subdirectory). The processor will recursively discover `**/*.pdf`.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000/rag`. Interactive docs are at `http://localhost:8000/rag/docs`.

---

## API Endpoints

All routes are prefixed with `/rag` (configurable via `APP_ROOT_PATH`).

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check — returns `{"status": "ok"}` |

---

### Data Processor — `/data_processor`

#### `POST /data_processor/process`

Load, clean, chunk, and optionally persist all PDFs found in `PDF_DATA_DIR`.

**Request body:**

```json
{
  "split": true,
  "parallel": true,
  "glob_pattern": "**/*.pdf",
  "max_workers": null,
  "persist": true,
  "persist_dir": null,
  "persist_format": "jsonl"
}
```

**Response:**

```json
{
  "total_files": 3,
  "total_pages": 120,
  "total_chunks": 450,
  "processed_files": ["data/doc1.pdf", "data/doc2.pdf"],
  "processed_at": "2026-04-27T10:00:00",
  "persisted": true,
  "persist_dir": "data/processed_chunks"
}
```

---

### Vector Store — `/load_vector_store`

#### `GET /load_vector_store/initialize_load_vector_store`

Loads persisted chunks from `data/processed_chunks/` and adds them to the FAISS index with Gemini embeddings. Embedding results are disk-cached under `vector_store/cache/`.

**Response:**

```json
{ "len_documents": 450 }
```

---

### Knowledge Graph — `/knowledge_graph`

#### `POST /knowledge_graph/build`

Loads persisted chunks, extracts `(subject, predicate, object)` triples using Gemini 2.5 Flash (rate-limited to ~10 req/min), streams triples to `data/processed_triples/triples.jsonl`, then ingests them into Neo4j using MERGE.

**Response:**

```json
{
  "total_documents": 450,
  "streamed_triples_written": 1820,
  "loaded_triples": 1820,
  "ingested_triples": 1540
}
```

#### `GET /knowledge_graph/query?cypher=<query>`

Execute an arbitrary Cypher query against Neo4j.

**Example:**

```
GET /knowledge_graph/query?cypher=MATCH (e:Entity)-[r:RELATION]->(o:Entity) RETURN e.name, r.predicate, o.name LIMIT 10
```

---

### Hybrid Fusion — `/hybrid_fusion`

#### `POST /hybrid_fusion/fuse`

Performs vector similarity search and knowledge graph traversal in parallel, then assembles a unified context block for downstream LLM usage.

**Request body:**

```json
{
  "query": "What are the side effects of aspirin?",
  "k": 4,
  "cypher": null,
  "include_scores": true
}
```

**Response:**

```json
{
  "query": "What are the side effects of aspirin?",
  "k": 4,
  "vector_hits_count": 4,
  "vector_hits": [...],
  "graph_results_count": 12,
  "graph_results": [...],
  "context": "User Query: ...\n=== Vector Similarity Hits ===\n..."
}
```

The `context` field is a pre-formatted string ready to be injected into an LLM prompt.

---

## Typical Usage Flow

```
1. POST /data_processor/process          ← Ingest PDFs → chunks persisted to disk
2. GET  /load_vector_store/initialize_load_vector_store  ← Build FAISS index
3. POST /knowledge_graph/build           ← Extract triples → build Neo4j graph
4. POST /hybrid_fusion/fuse              ← Query with hybrid retrieval
```

Steps 1–3 only need to run once (or when new documents are added). Step 4 can be called repeatedly for different queries.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | `""` | Google Gemini API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `""` | Neo4j password |
| `APP_ROOT_PATH` | `/rag` | FastAPI root path prefix |
| `PDF_DATA_DIR` | `data` | Directory scanned for PDFs |
| `PDF_CHUNK_SIZE` | `1000` | Chunk size in tokens |
| `PDF_CHUNK_OVERLAP` | `200` | Token overlap between chunks |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Gemini embedding model |
| `EMBEDDING_DIMENSION` | `768` | Embedding vector dimension |
| `VECTOR_STORE_PATH` | `vectorstore_db` | FAISS persistence directory |
| `VECTOR_SEARCH_K` | `4` | Default top-k for similarity search |
| `LLM_MODEL` | `gemini-pro` | Gemini chat/generation model |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FILE` | `logs/app.log` | Log file path |

---

## Running Tests

```bash
# Demo scripts (integration style)
python tests/demo_pdf_processor.py
python tests/demo_vector_store_service.py
python tests/demo_knowledge_graph_service.py
python tests/demo_hybrid_fusion_service.py

# Unit tests
pytest tests/
```

---

## License

For internal / personal use.
