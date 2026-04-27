# Developer Instructions — Graph + Vector RAG Project

This document provides detailed guidance for developers working on or extending this codebase. It covers the data pipeline, service patterns, configuration, API conventions, and how to add new capabilities.

---

## Table of Contents

1. [Development Environment Setup](#1-development-environment-setup)
2. [Project Conventions](#2-project-conventions)
3. [Configuration System](#3-configuration-system)
4. [Data Pipeline Deep Dive](#4-data-pipeline-deep-dive)
5. [Service Layer Patterns](#5-service-layer-patterns)
6. [Adding New Endpoints](#6-adding-new-endpoints)
7. [Embedding Cache](#7-embedding-cache)
8. [Neo4j Knowledge Graph](#8-neo4j-knowledge-graph)
9. [Hybrid Fusion Logic](#9-hybrid-fusion-logic)
10. [Logging](#10-logging)
11. [Testing](#11-testing)
12. [Common Troubleshooting](#12-common-troubleshooting)

---

## 1. Development Environment Setup

### Requirements

- Python 3.10 or higher
- Neo4j 5.x (local Desktop or Docker)
- Google AI Studio API key — enable **Gemini Embedding** and **Gemini 2.5 Flash**

### Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment File

Create `.env` in the project root before starting the server. At minimum:

```env
GOOGLE_API_KEY=<your key>
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your password>
```

All other values have safe defaults defined in `app/core/config.py`.

### Running Locally

```powershell
uvicorn app.main:app --reload
```

The server starts on `http://localhost:8000`. All routes are prefixed with `/rag` (the `APP_ROOT_PATH` setting). Swagger UI: `http://localhost:8000/rag/docs`.

---

## 2. Project Conventions

### Singleton Services

All three services (`PDFProcessor`, `VectorStoreService`, `KnowledgeGraphService`, `HybridFusionService`) follow the singleton pattern. Never instantiate them directly in production code; always call the factory function:

```python
# Correct
from app.services.data_processor import get_pdf_processor
processor = get_pdf_processor()

from app.services.vector_store_service import get_vector_store_service
vector_service = await get_vector_store_service()   # async
```

The factory functions cache the instance in a module-level variable. This prevents multiple Neo4j connections, duplicate FAISS indexes, and redundant embedding model initializations.

### Async vs Sync

| Layer | Style | Notes |
|---|---|---|
| Routers | `async def` | FastAPI handles the event loop |
| Services | `async def` | Neo4j async driver; FAISS ops wrapped in `asyncio.to_thread` |
| `PDFProcessor` | Sync | CPU-bound PDF I/O; called from routers via `Depends` (FastAPI runs sync deps in threadpool automatically) |
| LLM calls in KG service | `asyncio.to_thread(self._llm.invoke, ...)` | LangChain's `invoke` is synchronous |

Never call blocking I/O directly inside an `async def` without `asyncio.to_thread`.

### Error Handling

Routers catch all exceptions and re-raise as `HTTPException(status_code=500, detail=str(e))`. Services log errors at `ERROR` level and re-raise. Do not swallow exceptions silently.

---

## 3. Configuration System

`app/core/config.py` uses **pydantic-settings** (`BaseSettings`). Settings are loaded from:

1. `.env` file at project root (highest priority)
2. Real OS environment variables
3. Defaults declared in the `Field(default=...)` annotation

Access settings anywhere:

```python
from app.core.config import settings

print(settings.GOOGLE_API_KEY)
print(settings.NEO4J_URI)
```

`settings` is a **cached singleton** via `@lru_cache` — it is constructed once at import time. To add a new config value, add a `Field(...)` to the `Settings` class; it is immediately available via `settings.<NAME>`.

---

## 4. Data Pipeline Deep Dive

### Stage 1 — PDF Processing (`PDFProcessor`)

**File:** `app/services/data_processor.py`

Key methods:

| Method | Description |
|---|---|
| `load_all_pdfs()` | Sequential load via `DirectoryLoader` + `PyPDFLoader` |
| `load_all_pdfs_parallel()` | Multi-process load via `ProcessPoolExecutor` |
| `split_documents()` | Applies `RecursiveCharacterTextSplitter` (token-based via tiktoken) |
| `process_all_pdfs()` | Orchestrates load → clean → split → optional persist |
| `persist_chunks()` | Serializes `List[Document]` to JSONL at `data/processed_chunks/chunks.jsonl` |
| `load_persisted_chunks()` | Deserializes from JSONL back to `List[Document]` |

**Chunking strategy:** Token-boundary splitting using tiktoken model `text-embedding-3-small`. Default: 1000 tokens per chunk, 200 token overlap. This is set in `PDFProcessor.__init__` and controlled by `PDF_CHUNK_SIZE` / `PDF_CHUNK_OVERLAP` env vars.

**Text cleaning pipeline** (applied per page before splitting):
1. Replace non-breaking spaces (`\xa0`) with regular spaces
2. Collapse 3+ consecutive newlines into two
3. Convert single mid-sentence newlines to spaces
4. Collapse multiple spaces
5. Remove whitespace before punctuation

**Persistence format:** JSONL — one JSON object per line containing `page_content` and `metadata`. This is the interchange format between the processor and both the vector store and knowledge graph services.

---

### Stage 2 — Vector Store (`VectorStoreService`)

**File:** `app/services/vector_store_service.py`

1. Reads `data/processed_chunks/chunks.jsonl` via `processor.load_persisted_chunks()`
2. Initializes `GoogleGenerativeAIEmbeddings` (Gemini) wrapped in `CacheBackedEmbeddings`
3. Calls `FAISS.from_documents()` (or `add_texts()` to extend an existing index)
4. All FAISS operations run in `asyncio.to_thread` to keep the event loop unblocked

**Persistence:** The FAISS index and `InMemoryDocstore` are saved under `VECTOR_STORE_PATH`. On the next startup, they can be loaded from disk (not yet auto-loaded — must call the endpoint again or extend the startup hook).

---

### Stage 3 — Knowledge Graph (`KnowledgeGraphService`)

**File:** `app/services/knowledge_graph_service.py`

1. Loads persisted chunks
2. For each chunk, sends a structured prompt to Gemini 2.5 Flash requesting a JSON array of `{subject, predicate, object}` triples
3. Triples are **streamed to disk** (`data/processed_triples/triples.jsonl`) as they arrive — progress is preserved if the process is interrupted
4. A 6-second sleep between API calls keeps the request rate at ~10 req/min (free-tier limit)
5. After all chunks are processed, triples are loaded from disk and ingested into Neo4j using MERGE semantics

**Neo4j data model:**

```cypher
(:Entity {name: "subject"}) -[:RELATION {predicate: "predicate"}]-> (:Entity {name: "object"})
```

Provenance metadata (source PDF, page number) is attached to the `RELATION` relationship.

**Rate limiting:** The `await asyncio.sleep(6)` in `extract_triples` is intentional. Do not remove it without upgrading to a paid Gemini tier or implementing exponential backoff.

---

### Stage 4 — Hybrid Fusion (`HybridFusionService`)

**File:** `app/services/hybrid_fusion_service.py`

The `fuse()` method:

1. **Vector search** — calls `VectorStoreService.similarity_search_with_score(query, k=k)`
2. **Entity extraction** — heuristically extracts capitalized tokens from vector hits as entity candidates
3. **KG entity reconciliation** — fetches all existing entity names from Neo4j and intersects with candidates (currently returns all KG entities)
4. **Cypher generation** — if no custom Cypher provided, generates a parameterized `MATCH` query over the entity candidates
5. **Context assembly** — formats vector hits + graph rows into a structured text block

The `context` string in the response is designed to be directly concatenated into an LLM prompt.

---

## 5. Service Layer Patterns

### Adding a New Service

1. Create `app/services/my_service.py`
2. Implement the service class with an async `get_instance()` classmethod or a module-level factory function
3. Add the module-level singleton variable:

```python
_my_service_instance: Optional["MyService"] = None

async def get_my_service() -> "MyService":
    global _my_service_instance
    if _my_service_instance is None:
        _my_service_instance = MyService()
    return _my_service_instance
```

4. Use `SingletonLogger` for all log output:

```python
from app.core.logger import SingletonLogger
logger = SingletonLogger().get_logger()
```

---

## 6. Adding New Endpoints

1. Create a router file in `app/routers/my_router.py`:

```python
from fastapi import APIRouter, HTTPException, Depends
from app.services.my_service import get_my_service, MyService

router = APIRouter(prefix="/my_feature", tags=["my_feature"])

async def get_service() -> MyService:
    return await get_my_service()

@router.post("/action")
async def my_action(service: MyService = Depends(get_service)):
    try:
        result = await service.do_something()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. Register the router in `app/main.py`:

```python
from app.routers import my_router
app.include_router(my_router.router)
```

3. Add request/response Pydantic models to `app/models/request_models.py`.

---

## 7. Embedding Cache

The `VectorStoreService` uses LangChain's `CacheBackedEmbeddings` with a `LocalFileStore` at `vector_store/cache/`. Cached entries are keyed by `(namespace, text_hash)`.

**Benefits:** Re-running `initialize_load_vector_store` on the same corpus does not re-call the Gemini Embedding API.

**Clearing the cache:** Delete or rename the `vector_store/cache/` directory.

**Disabling the cache:** Set `use_cache=False` when constructing `VectorStoreService`, or modify the `_init_embeddings` method.

---

## 8. Neo4j Knowledge Graph

### Neo4j Connection

The async driver is initialized in `KnowledgeGraphService.__init__`:

```python
self._driver = AsyncGraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
)
```

Always call `await service.close()` during application shutdown to release the connection pool.

### Querying the Graph

Use the `/knowledge_graph/query` endpoint for ad-hoc exploration, or call `service.query(cypher, params)` directly from code.

**Useful Cypher queries:**

```cypher
-- Count all entities
MATCH (e:Entity) RETURN count(e)

-- List all unique predicates
MATCH ()-[r:RELATION]->() RETURN DISTINCT r.predicate ORDER BY r.predicate

-- Find all facts about a specific entity
MATCH (e:Entity {name: "Aspirin"})-[r:RELATION]->(o:Entity)
RETURN e.name, r.predicate, o.name

-- Find paths between two entities
MATCH p = shortestPath(
  (a:Entity {name: "Aspirin"})-[*]-(b:Entity {name: "COX-2"})
) RETURN p
```

### Rebuilding the Graph

The build process **appends** new triples to `triples.jsonl` and uses `MERGE` in Neo4j, so re-running `POST /knowledge_graph/build` is idempotent — it will not duplicate nodes or relationships. To start fresh, delete `data/processed_triples/triples.jsonl` and run `MATCH (n) DETACH DELETE n` in Neo4j Browser first.

---

## 9. Hybrid Fusion Logic

### Custom Cypher in Fusion

Pass a `cypher` field in the `/hybrid_fusion/fuse` request to override the auto-generated query:

```json
{
  "query": "aspirin side effects",
  "k": 4,
  "cypher": "MATCH (e:Entity)-[r:RELATION]->(o:Entity) WHERE e.name CONTAINS 'Aspirin' RETURN e.name AS subject, r.predicate AS predicate, o.name AS object",
  "include_scores": true
}
```

### Extending the Context Builder

The `_build_context` method in `HybridFusionService` assembles the final string. Modify it to change the format passed to your LLM. The current structure is:

```
User Query: <query>

=== Vector Similarity Hits ===
[1] <snippet> (score=0.8123)
[2] ...

=== Knowledge Graph Relationships ===
<subject> -[<predicate>]-> <object>
...

=== Guidance ===
Use the entities and relationships above...
```

---

## 10. Logging

All application logging goes through the `SingletonLogger` (thread-safe, double-checked locking):

```python
from app.core.logger import SingletonLogger
logger = SingletonLogger().get_logger()

logger.info("Something happened")
logger.error("Something went wrong")
```

Logs are written to:
- **File:** `logs/app.log` (rotating, max 5 MB, 5 backups)
- **Console:** stdout

**Per-request logging** is handled by `LoggingMiddleware` in `app/core/logging_middleware.py`, which records method, URL, status code, and latency for every HTTP request.

---

## 11. Testing

The `tests/` directory contains both demo/integration scripts and unit tests.

### Demo scripts (run directly)

```powershell
# Test PDF processing only
python tests/demo_pdf_processor.py

# Test vector store build + query
python tests/demo_vector_store_service.py

# Test triple extraction + Neo4j ingestion
python tests/demo_knowledge_graph_service.py

# Test hybrid fusion end-to-end
python tests/demo_hybrid_fusion_service.py

# Benchmark parallel vs sequential PDF loading
python tests/benchmark_parallel.py
```

### Pytest unit tests

```powershell
pytest tests/
```

### Calling the live API

```powershell
# Health check
curl http://localhost:8000/rag/health

# Process PDFs
curl -X POST http://localhost:8000/rag/data_processor/process `
  -H "Content-Type: application/json" `
  -d '{"split": true, "parallel": true, "persist": true}'

# Build vector index
curl http://localhost:8000/rag/load_vector_store/initialize_load_vector_store

# Build knowledge graph
curl -X POST http://localhost:8000/rag/knowledge_graph/build

# Hybrid fusion query
curl -X POST http://localhost:8000/rag/hybrid_fusion/fuse `
  -H "Content-Type: application/json" `
  -d '{"query": "What are the main findings?", "k": 4}'
```

---

## 12. Common Troubleshooting

### `GOOGLE_API_KEY missing for Gemini embeddings`

The `.env` file is missing or `GOOGLE_API_KEY` is empty. Verify:

```powershell
Get-Content .env | Select-String GOOGLE_API_KEY
```

### Neo4j connection refused

- Confirm Neo4j is running: open Neo4j Browser at `http://localhost:7474`
- Confirm `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` match your Neo4j instance in `.env`

### `No PDF files found`

`PDF_DATA_DIR` defaults to `data`. Ensure your PDFs are inside the `data/` folder (subdirectories are fine — the processor uses `**/*.pdf`).

### Triple extraction returns 0 triples

- Gemini 2.5 Flash returned non-JSON for all chunks. Check `logs/app.log` for `Non-JSON response` debug messages.
- The model may be rate-limited. The 6-second sleep between calls handles 10 req/min. If you see 429 errors, increase the sleep duration in `KnowledgeGraphService.extract_triples`.

### `chunks.jsonl not found` when calling `/load_vector_store` or `/knowledge_graph/build`

Run `POST /data_processor/process` with `"persist": true` first.

### FAISS index out of date after adding new documents

Call `GET /load_vector_store/initialize_load_vector_store` again. The service will add the new chunks to the existing index (if already initialized) or build a fresh one.

### Port conflict on startup

If port 8000 is already in use:

```powershell
uvicorn app.main:app --reload --port 8001
```

Update any API calls to use the new port accordingly.
