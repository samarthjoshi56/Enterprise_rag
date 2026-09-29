# Enterprise RAG Platform — Comprehensive Technical Report

> **Project:** Enterprise Retrieval-Augmented Generation (RAG) System
> **Stack:** Python 3.13 · FastAPI · PostgreSQL 16 · Qdrant · Redis · React 19 · TypeScript · Vite · Tailwind CSS
> **Tests:** 70 passing across all phases
> **API:** 20+ REST endpoints
> **UI:** 7 fully functional sections

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Phase-by-Phase Implementation](#4-phase-by-phase-implementation)
5. [Database Schema](#5-database-schema)
6. [API Reference](#6-api-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [UI Usage Guide](#8-ui-usage-guide)
9. [Running the Project Locally](#9-running-the-project-locally)
10. [Project File Structure](#10-project-file-structure)

---

## 1. What Is This Project?

The **Enterprise RAG Platform** is a production-grade, end-to-end Retrieval-Augmented Generation system designed for internal enterprise knowledge management. It allows an organisation to:

- **Ingest** arbitrary document collections (PDF, TXT) and store them as semantically searchable vector embeddings.
- **Search** across those documents using a sophisticated hybrid pipeline combining dense vector search with traditional keyword full-text search, fused and reranked by a cross-encoder neural model.
- **Chat** with the document corpus in a conversational interface backed by retrieved context.
- **Query structured data** in natural language via a Text2SQL pipeline with human-in-the-loop approval before any SQL is executed.
- **Cache** expensive retrieval computations in Redis for sub-millisecond repeat query response.
- **Evaluate** retrieval quality using Precision@K, Recall@K, MRR, and Hit Rate — and benchmark multiple retrieval configurations side-by-side.

The system is **not** a simple wrapper around a single LLM call. It implements multiple research techniques from recent NLP literature:

- **HyDE** (Gao et al., 2022) — Hypothetical Document Embeddings
- **CRAG** (Yan et al., 2024) — Corrective RAG with quality grading
- **Self-RAG** (Asai et al., 2023) — Self-reflection tokens for retrieval decisions

All three techniques are orchestrated in a stateful **LangGraph** workflow, sitting on top of a dual-database storage layer (Qdrant for vectors, PostgreSQL for structured metadata and full-text search).

The frontend is a full React SPA with a dark glassmorphism design that consumes every backend API and presents a complete operator-facing control panel.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Vite)                       │
│  Dashboard · Documents · Chat · Search · Text2SQL · Eval · Settings │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (proxied via Vite)
┌──────────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend (Uvicorn)                        │
│                         /api/v1/*                                    │
│                                                                      │
│  ┌─────────────┐  ┌────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Documents  │  │   Search   │  │  Text2SQL │  │  Evaluation  │  │
│  │   /upload   │  │  /search   │  │ /generate │  │    /run      │  │
│  │   /list     │  │            │  │ /execute  │  │  /compare    │  │
│  └──────┬──────┘  └─────┬──────┘  └─────┬─────┘  └──────────────┘  │
│         │               │               │                            │
│  ┌──────▼──────┐  ┌─────▼──────────────▼────────────────────────┐  │
│  │  Ingestion  │  │             Search Service                   │  │
│  │   Pipeline  │  │  ┌──────────┐  ┌─────────┐  ┌──────────┐   │  │
│  │  Loader     │  │  │  Vector  │  │ Keyword │  │ Reranker │   │  │
│  │  Chunker    │  │  │  Search  │  │  FTS    │  │  Cross-  │   │  │
│  │  Embeddings │  │  │  Qdrant  │  │ Postgres│  │  Encoder │   │  │
│  └──────┬──────┘  │  └──────────┘  └─────────┘  └──────────┘   │  │
│         │         │         ↑ RRF Fusion ↑                       │  │
│         │         │  ┌─────────────────────────────────────────┐ │  │
│         │         │  │         Advanced RAG (LangGraph)         │ │  │
│         │         │  │   HyDE  →  CRAG  →  Self-RAG  →  END    │ │  │
│         │         │  └─────────────────────────────────────────┘ │  │
│         │         │         ↑ Redis Cache Layer ↑                │  │
│         │         └─────────────────────────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
│                                                                      │
│  ┌───────────────────────┐    ┌─────────────────────────────────┐   │
│  │  PostgreSQL 16         │    │  Qdrant (Vector DB)             │   │
│  │  documents             │    │  Collection: enterprise_docs    │   │
│  │  document_chunks (FTS) │    │  384-dim dense vectors          │   │
│  │  incidents             │    │  Payload: chunk metadata        │   │
│  └───────────────────────┘    └─────────────────────────────────┘   │
│  ┌───────────────────────┐                                           │
│  │  Redis 7               │                                           │
│  │  TTL-based query cache │                                           │
│  │  Prefix: rag:cache:    │                                           │
│  └───────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow — A Search Query

```
User query
  │
  ▼ Check Redis cache (normalized SHA-256 key)
  │  HIT  → return cached result immediately (sub-ms)
  │  MISS ↓
  ▼ Vector search (Qdrant ANN, top-K=20)
  + Keyword search (PostgreSQL FTS, top-K=20)
  │
  ▼ Reciprocal Rank Fusion (RRF) + alpha weighting
  │  default alpha=0.7: 70% vector weight, 30% keyword weight
  │
  ▼ Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
  │  Reranks all candidates, returns top-N=5
  │
  ▼ Store result in Redis (TTL=3600s)
  │
  ▼ Return structured JSON response
```

---

## 3. Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web framework | **FastAPI** | ≥0.110 | Async REST API |
| Server | **Uvicorn** | ≥0.28 | ASGI server |
| Data validation | **Pydantic v2** | ≥2.6 | Request/response schemas |
| ORM | **SQLAlchemy** | ≥2.0 | Async DB sessions |
| Async PG driver | **asyncpg** | ≥0.29 | PostgreSQL async |
| Sync PG driver | **psycopg2** | ≥2.9 | DB utilities |
| Vector DB client | **qdrant-client** | ≥1.8 | Qdrant HTTP + gRPC |
| Cache client | **redis-py** | ≥5.0 | Redis async |
| PDF parsing | **pypdf** | ≥4.0 | Text + page extraction |
| Embeddings | **sentence-transformers** | ≥2.5 | all-MiniLM-L6-v2 (384d) |
| Reranker | **sentence-transformers** | ≥2.5 | ms-marco-MiniLM-L-6-v2 |
| LLM orchestration | **LangChain** | ≥0.3 | HyDE/CRAG/Self-RAG prompts |
| Graph workflow | **LangGraph** | ≥0.2 | Stateful RAG graph |
| LLM backend | **OpenAI GPT-4o-mini** | — | Text generation |
| SQL parsing | **sqlparse** | ≥0.5 | SQL safety validation |
| Testing | **pytest + pytest-asyncio** | ≥8.0 | 70 async tests |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Build tool | **Vite** | 8.x | Dev server + bundler |
| Framework | **React** | 19.x | UI component library |
| Language | **TypeScript** | 6.x | Type-safe development |
| Styling | **Tailwind CSS** | 4.x | Utility CSS |
| Routing | **react-router-dom** | 7.x | Client-side routing |
| Icons | **lucide-react** | latest | SVG icon set |
| Charts | **recharts** | latest | Radar + bar charts |
| HTTP | Native fetch | — | Proxied via Vite |

### Infrastructure

| Service | Version | Port | Purpose |
|---------|---------|------|---------|
| PostgreSQL | 16 | 5432 | Structured metadata, FTS |
| Qdrant | 1.11 | 6333/6334 | Vector storage and ANN search |
| Redis | 7 | 6379 | Query result caching |

---

## 4. Phase-by-Phase Implementation

### Phase 1 — Infrastructure Setup

**Goal:** Establish the foundational project skeleton, configuration system, database connections, and health monitoring.

**Key files:**

- `app/core/config.py` — Pydantic Settings with `.env` support. All 30+ config parameters derived from environment variables with sensible defaults. Singleton via `@lru_cache()`.
- `app/db/postgres.py` — SQLAlchemy `AsyncEngine` + session factory. `init_postgres_db()` creates all tables on startup.
- `app/db/qdrant.py` — Async Qdrant client. `init_qdrant_collection()` creates the `enterprise_documents` collection with cosine distance if it doesn't exist.
- `app/db/redis.py` — Async Redis client via `redis.asyncio`.
- `app/api/health.py` — `GET /health` concurrently checks all three services and reports individual latencies, Qdrant version, collection counts, and LangChain/LangGraph framework versions.
- `app/main.py` — FastAPI lifespan handler; CORS middleware with `allow_origins=["*"]`; routes registered at `/api/v1/` and root level.

**Design decisions:**

- Settings cached via `@lru_cache()` — zero re-parsing overhead for the process lifetime.
- Health endpoint returns HTTP 200 even if a service is degraded (for inspection). Returns HTTP 503 only if ALL services are unreachable.
- CORS is wide open (`*`) for local development — tighten in production.

---

### Phase 2 — Document Ingestion

**Goal:** Accept PDF and TXT files, extract text with page metadata, split into overlapping chunks, embed each chunk, and persist to both Qdrant and PostgreSQL.

**Pipeline:**

```
File Upload (multipart/form-data)
  │
  ▼ Loader (app/ingestion/loader.py)
  │  PDF  → pypdf PdfReader, page-by-page text extraction
  │  TXT  → raw UTF-8 decode
  │  Output: List[{"text": ..., "page_number": ..., "filename": ...}]
  │
  ▼ Chunker (app/ingestion/chunker.py)
  │  LangChain RecursiveCharacterTextSplitter
  │  CHUNK_SIZE=1000, CHUNK_OVERLAP=200 (configurable via .env)
  │  Output: List[Chunk] with chunk_index, page_number, text
  │
  ▼ Embeddings (app/ingestion/embeddings.py)
  │  SentenceTransformer("all-MiniLM-L6-v2")
  │  384-dimensional dense vectors, normalize_embeddings=True
  │
  ▼ Qdrant Storage
  │  Collection: enterprise_documents
  │  Payload per point: chunk_id, document_id, filename,
  │                     page_number, chunk_index, text
  │
  ▼ PostgreSQL Storage
     Table: documents (document-level metadata, status tracking)
     Table: document_chunks (chunk-level, chunk_text for FTS)
```

**Key implementation details:**

- Each `Document` record tracks `status`: PENDING → PROCESSING → COMPLETED | FAILED, with `error_message` for debugging.
- Chunk text is stored in **both** Qdrant (as payload for retrieval) and PostgreSQL (as `chunk_text` for FTS).
- Qdrant point UUIDs are stored back in PostgreSQL as `qdrant_point_id` for cross-referencing.
- After every upload, the Redis cache is automatically flushed — stale search results are always evicted.

---

### Phase 3 — Hybrid Search

**Goal:** Given a natural-language query, return the most relevant document chunks using a multi-stage retrieval pipeline.

**Pipeline:**

```
Query string
  │
  ├── [Vector Search]  app/search/vector_search.py
  │    Embed query → 384d vector via SentenceTransformer
  │    Qdrant query_points (ANN, cosine similarity)
  │    Returns top VECTOR_SEARCH_TOP_K=20 hits with scores
  │
  ├── [Keyword Search]  app/search/keyword_search.py
  │    PostgreSQL FTS: to_tsvector(chunk_text) + to_tsquery(query)
  │    ts_rank scoring
  │    Returns top KEYWORD_SEARCH_TOP_K=20 results
  │
  ├── [Hybrid Fusion]  app/search/hybrid.py
  │    Reciprocal Rank Fusion (RRF):
  │      rrf_score = sum(1 / (60 + rank_i))
  │    Alpha weighting:
  │      hybrid_score = alpha * vector_rrf + (1-alpha) * keyword_rrf
  │    Default alpha=0.7 | Deduplication by chunk_id
  │
  └── [Reranking]  app/search/reranker.py
       cross-encoder/ms-marco-MiniLM-L-6-v2
       Scores (query, chunk_text) pairs via cross-attention
       Returns top RERANKER_TOP_N=5 results
```

**Why both vector AND keyword?**
Vector search excels at semantic similarity but struggles with exact term matching, acronyms, and proper nouns. Keyword search is exact but not semantic. RRF fusion captures the best of both worlds with a single configurable blend weight.

**Response fields per chunk:** `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_index`, `text`, `vector_score`, `keyword_score`, `hybrid_score`, `rerank_score`, `latency_ms`, `cached`, `total_candidates`, `vector_hits`, `keyword_hits`.

---

### Phase 4 — Advanced RAG

**Goal:** Augment hybrid retrieval with three research-backed techniques in a LangGraph stateful graph.

#### HyDE — Hypothetical Document Embeddings (Gao et al., 2022)

**File:** `app/rag/hyde.py`

Instead of embedding the query directly, HyDE:
1. Prompts an LLM to generate a *hypothetical answer* to the query.
2. Embeds the hypothetical answer — it lives in the answer space, closer to real corpus answers.
3. Searches Qdrant using that embedding.

Comparison metrics (Jaccard similarity) quantify how much HyDE retrieval differs from standard retrieval.

#### CRAG — Corrective RAG (Yan et al., 2024)

**File:** `app/rag/crag.py`

Grades retrieved chunks using the cross-encoder model:

```
Grade == CORRECT   → use chunks as-is
Grade == AMBIGUOUS → keep chunks, log warning
Grade == INCORRECT → rewrite query, execute corrective search
```

Relevance threshold (`CRAG_RELEVANCE_THRESHOLD=-2.0`) is a cross-encoder logit. Configurable in `.env`.

#### Self-RAG (Asai et al., 2023)

**File:** `app/rag/self_rag.py`

Adds self-reflection tokens:
- **Retrieve?** — should retrieval even happen for this query?
- **IsREL** — are retrieved chunks actually relevant?
- **IsSUP** — is a generated response supported by context (hallucination detection)?

#### LangGraph Workflow

**File:** `app/rag/graph.py`

All three techniques in a directed graph:

```
START
  ▼
node_query          (Self-RAG: should_retrieve? YES/NO)
  ▼
node_hybrid_retrieve  (Standard hybrid search + HyDE in parallel)
  ▼
node_evaluate_relevance  (CRAG grading + Self-RAG IsREL)
  │
  ├── CORRECT / max retries reached → node_final_context (Rerank + return)
  └── INCORRECT → node_correct_retrieval (CRAG rewrite + re-retrieve)
                        │
                        └──────────────────→ node_final_context
```

Graph state TypedDict: `query`, `candidates`, `retries`, `grade`, `hypo_doc`, `final_chunks`.

---

### Phase 5 — Text2SQL

**Goal:** Natural language → SQL with mandatory human approval before any execution.

**Pipeline:**

```
Natural language question
  │
  ▼ Schema context injection (app/text2sql/schema.py)
  │  Allowed tables: documents, document_chunks, incidents
  │  Column descriptions injected into LLM system prompt
  │
  ▼ SQL generation (app/text2sql/generator.py)
  │  LangChain LLMChain → generated_sql + explanation
  │
  ▼ SQL validation (app/text2sql/validator.py)
  │  sqlparse tokenization → reject if NOT a SELECT
  │  Block list: DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE
  │  SQL injection pattern detection
  │  Returns is_safe + error message
  │
  ▼ Staging (app/text2sql/approval.py)
  │  query_id (UUID) in in-memory registry
  │  status = "pending_approval"
  │
  ▼ Human Approval (via UI or API)
  │  approved=True  → execute
  │  approved=False → status = "rejected"
  │
  ▼ Execution (app/text2sql/executor.py)
     SET TRANSACTION READ ONLY (belt-and-suspenders safety)
     SQLAlchemy raw SQL execution
     Returns columns[], rows[], row_count, execution_time_ms
```

**Safety layers:**
1. Validator syntactically rejects any non-SELECT statement.
2. Execution session uses `SET TRANSACTION READ ONLY` — the database prevents mutations even if the validator is bypassed.
3. Only approved tables are exposed in the schema context the LLM sees.

---

### Phase 6 — Redis Caching

**Goal:** Cache hybrid search results to eliminate redundant embedding + ANN + FTS + reranking work for repeated queries.

**Cache key:**

```python
cache_key = sha256(normalize(query) + "::" + params_fingerprint)
# normalize: lowercase, collapse whitespace, strip
```

`"  What is Kubernetes?  "` and `"what is kubernetes?"` produce the same key.

**Integration:**
- `app/search/service.py` — checks Redis before pipeline; writes to Redis after.
- `app/ingestion/service.py` — calls `cache_service.clear()` after every document ingestion.
- `app/cache/service.py` — async `get()`, `set()`, `delete()`, `clear(prefix)`, `get_stats()`.

**TTL:** Default 3600s (1 hour), configurable via `REDIS_CACHE_TTL`.

---

### Phase 7 — Evaluation Framework

**Goal:** Objectively measure retrieval quality and compare pipeline configurations.

**Metrics** (`app/evaluation/metrics.py`):

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision@K** | `relevant_in_top_k / K` | How precise are the top-K results? |
| **Recall@K** | `relevant_in_top_k / total_relevant` | How many relevant docs were found? |
| **MRR** | `mean(1 / rank_of_first_hit)` | How early does the first relevant result appear? |
| **Hit Rate** | `queries_with_any_hit / total_queries` | What fraction of queries return at least one hit? |

**Comparison benchmark** (`app/evaluation/comparison.py`) — 5 configurations:

1. Vector-only
2. Keyword-only
3. Hybrid (no reranking)
4. Hybrid + Reranking (full pipeline)
5. Advanced RAG (HyDE + CRAG + Self-RAG + Reranking)

---

### Phase 8 — React Frontend UI

**Goal:** A production-quality, dark-themed React SPA that consumes every backend API.

**Technical decisions:**

- **Vite proxy** — all `/api` calls proxied to `http://127.0.0.1:8000`. Zero CORS config needed.
- **Native fetch** — typed wrappers in `src/api/client.ts`, TypeScript interfaces matching every Pydantic schema.
- **No state library** — each page self-contained with `useState` + `useEffect`.
- **Tailwind CSS v4** — via `@tailwindcss/vite` plugin. No PostCSS config required.
- **Custom design system** — CSS variables for full color palette, glass cards, animations, score bars.

**Design system color tokens:**

```css
--bg-base:          #0a0d14   /* page background */
--bg-surface:       #111827   /* sidebar */
--bg-elevated:      #1a2236   /* cards, inputs */
--accent-primary:   #6366f1   /* indigo */
--accent-secondary: #8b5cf6   /* purple */
--accent-tertiary:  #06b6d4   /* cyan */
--accent-success:   #10b981
--accent-warning:   #f59e0b
--accent-danger:    #ef4444
```

**Charts:** Recharts `RadarChart` (evaluation overview) and `BarChart` (config comparison).

---

## 5. Database Schema

### Table: `documents`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Unique document identifier |
| `filename` | VARCHAR(255) | Original file name |
| `file_type` | VARCHAR(50) | `pdf` or `txt` |
| `file_size_bytes` | INTEGER | Raw file size |
| `total_chunks` | INTEGER | Number of chunks produced |
| `total_pages` | INTEGER | Pages (PDF) or 1 (TXT) |
| `status` | VARCHAR(50) | PENDING / PROCESSING / COMPLETED / FAILED |
| `error_message` | TEXT | Failure reason if FAILED |
| `created_at` | TIMESTAMPTZ | Ingestion timestamp |
| `updated_at` | TIMESTAMPTZ | Last status update |

### Table: `document_chunks`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Unique chunk identifier |
| `document_id` | UUID (FK) | Parent document |
| `chunk_index` | INTEGER | Sequential position |
| `page_number` | INTEGER (nullable) | Source page (PDF only) |
| `qdrant_point_id` | UUID | Corresponding Qdrant point |
| `character_count` | INTEGER | Chunk text length |
| `chunk_text` | TEXT | Raw text (used for FTS) |
| `created_at` | TIMESTAMPTZ | Creation timestamp |

### Table: `incidents` (Text2SQL demo)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Incident ID |
| `title` | VARCHAR(255) | Incident title |
| `service` | VARCHAR(100) | Affected service |
| `severity` | VARCHAR(50) | LOW / MEDIUM / HIGH / CRITICAL |
| `status` | VARCHAR(50) | OPEN / RESOLVED |
| `impact_summary` | TEXT | Description |
| `created_at` | TIMESTAMPTZ | Incident time |
| `resolved_at` | TIMESTAMPTZ (nullable) | Resolution time |

### Qdrant Collection: `enterprise_documents`

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Point ID |
| vector | float32[384] | Dense embedding (cosine distance) |
| `chunk_id` | payload string | UUID matching PostgreSQL chunk |
| `document_id` | payload string | Parent document UUID |
| `filename` | payload string | Source filename |
| `page_number` | payload int | Source page |
| `chunk_index` | payload int | Position in document |
| `text` | payload string | Raw chunk text |

---

## 6. API Reference

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Full system health check |
| GET | `/health/liveness` | Kubernetes liveness probe |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/documents/upload` | Upload and ingest PDF/TXT |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/documents/{id}` | Document detail + chunks |

### Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/search` | Hybrid search pipeline |

Request body:

```json
{
  "query": "string (required)",
  "vector_top_k": 20,
  "keyword_top_k": 20,
  "alpha": 0.7,
  "top_n": 5,
  "bypass_cache": false
}
```

### Text2SQL

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/text2sql/schema` | View allowed tables |
| POST | `/api/v1/text2sql/generate` | Generate SQL — Step 1 |
| POST | `/api/v1/text2sql/execute` | Approve/reject — Step 2 |

### Cache

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/cache/stats` | Redis statistics |
| POST | `/api/v1/cache/clear` | Clear all or prefix-scoped keys |
| DELETE | `/api/v1/cache/keys/{key}` | Delete specific key |

### Evaluation

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/evaluation/dataset` | Benchmark dataset |
| POST | `/api/v1/evaluation/run` | Run retrieval evaluation |
| POST | `/api/v1/evaluation/compare` | Compare all 5 configurations |

---

## 7. Configuration Reference

All settings loaded from `.env` in the project root:

```env
# Application
APP_NAME=Enterprise RAG
APP_ENV=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=enterprise_rag

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Embedding and Chunking
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
QDRANT_COLLECTION_NAME=enterprise_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search and Retrieval
VECTOR_SEARCH_TOP_K=20
KEYWORD_SEARCH_TOP_K=20
HYBRID_ALPHA=0.7
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=5

# Advanced RAG
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini
HYDE_ENABLED=true
CRAG_RELEVANCE_THRESHOLD=-2.0
CRAG_MAX_REWRITES=1
SELF_RAG_ENABLED=true

# Caching
REDIS_CACHE_ENABLED=true
REDIS_CACHE_TTL=3600
REDIS_CACHE_PREFIX=rag:cache:

# Evaluation
EVALUATION_DATASET_PATH=data/evaluation_dataset.json
```

---

## 8. UI Usage Guide

Open **http://localhost:5173** in your browser with the backend running at port 8000.

### Dashboard

The landing page gives a real-time system overview:

- **Stat cards** — total documents, total chunks in Qdrant, overall system health.
- **Service Health panel** — live status dots for PostgreSQL, Qdrant, Redis with latency readings. Green pulsing dot = connected, red = error.
- **Recent Documents panel** — last 8 ingested documents with status icons.
- **Refresh button** — re-fetches health and document list concurrently.

### Documents

Upload and manage your knowledge base:

1. **Upload** — drag a PDF or TXT onto the upload zone, or click to browse. A progress bar animates while the backend parses, chunks, embeds, and stores the document.
2. **Document table** — file type badge, size, pages, chunk count, status badge, ingestion date.
3. **Detail drawer** — click the `→` button to open a slide-in panel showing full metadata and a scrollable chunk list (page number, character count).

> After upload, Redis cache is automatically invalidated so the next search includes the new document.

### RAG Chat

Chat with your document corpus:

1. Type a question and press **Enter** or click **Send**.
2. The system runs hybrid search + reranking and synthesizes an answer from the top 3 chunks.
3. Each response shows:
   - The synthesized answer with **bolded** source references.
   - A **Retrieved Sources** accordion — expand any card to read the raw chunk text.
   - Latency (ms) and a **Cached** badge if the result came from Redis.

### Search

Direct access to the full hybrid search pipeline:

1. Type a query and click **Search**.
2. Click the sliders icon to open advanced options:
   - **Alpha slider** — 0 = pure keyword, 1 = pure vector.
   - **Top-N slider** — number of reranked results returned (1–20).
   - **Bypass cache** — forces a fresh pipeline run.
3. **Result cards** — filename, page, rerank score, 2-line preview. Expand to see vector score, keyword score, and hybrid score with animated fill bars.

### Text2SQL

Query structured data with natural language:

1. Type a question (or click a quick example at the bottom).
2. Click **Generate SQL** — the LLM generates SQL, the validator checks it. You see:
   - Sanitized SQL in a syntax-highlighted code block.
   - Plain-English explanation.
   - Safe/Unsafe badge and status.
3. **Review the SQL carefully**, then:
   - **Approve and Run** — executes and shows results in a table.
   - **Reject** — discards the query.
4. Results show column headers, all rows, row count, and execution time.

> Only `SELECT` queries are ever allowed. Mutations are blocked at two independent layers.

### Evaluation

Benchmark retrieval quality:

**Run Evaluation tab:**
1. Set the K cutoff with the slider.
2. Click **Run** — backend runs the benchmark dataset through the hybrid pipeline.
3. Four metric cards (Precision@K, Recall@K, MRR, Hit Rate) with color-coded bars (green ≥70%, amber ≥40%, red <40%) and a Radar Chart overview.

**Compare Configs tab:**
1. Click **Compare** — runs all 5 retrieval configurations against the benchmark.
2. A grouped Bar Chart and comparison table with score bars per metric.

**Dataset tab:**
1. Click **Load** to see all benchmark questions with ground-truth keywords and target services.

### Settings

System administration:

- **Backend Services** — live status for all three services with versions.
- **Model Configuration** — embedding model, reranker model, LLM.
- **Framework Versions** — LangChain, LangGraph, langchain-core.
- **Redis Cache** — cached query count, memory usage, hit/miss ratio. **Clear All Cache** button.

---

## 9. Running the Project Locally

### Prerequisites

- Python 3.11+ (3.13 recommended)
- Node.js 20+
- PostgreSQL 16 on port 5432
- Qdrant on port 6333
- Redis 7 on port 6379

### Backend

```bash
cd "Enterprise rag"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, OPENAI_API_KEY

# Start
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Run tests
pytest tests/ -v
```

Swagger UI at **http://127.0.0.1:8000/docs**

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI at **http://localhost:5173**

### Starting Infrastructure (macOS + Homebrew)

```bash
brew services start postgresql@16
brew services start redis

# Qdrant via Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## 10. Project File Structure

```
Enterprise rag/
├── app/
│   ├── main.py                    FastAPI app, lifespan, CORS, routing
│   ├── core/
│   │   ├── config.py              Pydantic Settings (30+ env vars)
│   │   └── logging.py             Structured logging setup
│   ├── db/
│   │   ├── models.py              SQLAlchemy ORM: Document, DocumentChunk, Incident
│   │   ├── postgres.py            AsyncEngine, session factory, init
│   │   ├── qdrant.py              Async Qdrant client, collection init
│   │   └── redis.py               Async Redis client
│   ├── api/
│   │   ├── routes.py              APIRouter aggregator
│   │   ├── health.py              GET /health
│   │   ├── documents.py           Upload, list, get document
│   │   ├── search.py              POST /search (hybrid pipeline)
│   │   ├── text2sql.py            Generate + execute SQL endpoints
│   │   ├── cache.py               Cache stats + clear endpoints
│   │   └── evaluation.py          Run + compare evaluation endpoints
│   ├── ingestion/
│   │   ├── loader.py              PDF (pypdf) + TXT parsing
│   │   ├── chunker.py             RecursiveCharacterTextSplitter
│   │   ├── embeddings.py          SentenceTransformer all-MiniLM-L6-v2
│   │   └── service.py             End-to-end ingestion pipeline
│   ├── search/
│   │   ├── vector_search.py       Qdrant ANN search
│   │   ├── keyword_search.py      PostgreSQL FTS (to_tsvector)
│   │   ├── hybrid.py              RRF fusion + alpha weighting
│   │   ├── reranker.py            Cross-encoder ms-marco-MiniLM-L-6-v2
│   │   ├── models.py              ChunkResult dataclass
│   │   └── service.py             Pipeline orchestration + Redis cache
│   ├── rag/
│   │   ├── llm.py                 LangChain LLM chains (OpenAI)
│   │   ├── hyde.py                Hypothetical Document Embeddings
│   │   ├── crag.py                Corrective RAG (grader + rewriter)
│   │   ├── self_rag.py            Self-RAG reflection tokens
│   │   ├── graph.py               LangGraph stateful workflow
│   │   └── service.py             Advanced RAG entry point
│   ├── text2sql/
│   │   ├── schema.py              Allowed tables + column descriptions
│   │   ├── generator.py           LangChain SQL generation chain
│   │   ├── validator.py           sqlparse safety validation
│   │   ├── approval.py            In-memory HITL staging registry
│   │   ├── executor.py            Read-only SQL execution
│   │   └── service.py             Pipeline orchestration
│   ├── cache/
│   │   └── service.py             Redis cache get/set/clear/stats
│   └── evaluation/
│       ├── dataset.py             Benchmark dataset loader
│       ├── metrics.py             Precision@K, Recall@K, MRR, Hit Rate
│       ├── runner.py              Evaluation runner
│       └── comparison.py          5-config benchmark comparison
├── frontend/
│   ├── src/
│   │   ├── api/client.ts          Typed fetch wrappers for all endpoints
│   │   ├── components/
│   │   │   ├── Sidebar.tsx        Navigation sidebar
│   │   │   └── StatCard.tsx       Reusable metric card
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx      System overview
│   │   │   ├── Documents.tsx      Upload + document list
│   │   │   ├── Chat.tsx           RAG chat interface
│   │   │   ├── SearchPage.tsx     Hybrid search UI
│   │   │   ├── Text2SQL.tsx       NL to SQL with HITL
│   │   │   ├── Evaluation.tsx     Metrics + comparison charts
│   │   │   └── Settings.tsx       Config + cache management
│   │   ├── App.tsx                Router + layout
│   │   ├── main.tsx               React entry point
│   │   └── index.css              Design system (tokens, glass, animations)
│   ├── vite.config.ts             Vite + Tailwind + proxy config
│   └── package.json
├── tests/                         28 test files, 70 tests
├── scripts/                       CLI verification scripts
├── data/
│   └── evaluation_dataset.json    Benchmark questions + ground truth
├── docker-compose.yml             PostgreSQL + Qdrant + Redis
├── pyproject.toml                 Python dependencies + pytest config
├── REPORT.md                      This document
└── .env                           Environment variables
```

---

*This report was generated from the live codebase. All implementation details reflect the actual source code in this repository.*
