# Enterprise RAG - Phase 1: Project Setup & Core Infrastructure

Production-style architecture foundation for Enterprise RAG (Retrieval-Augmented Generation), featuring FastAPI, PostgreSQL, Qdrant, Redis, LangChain, and LangGraph.

---

## 🏗️ Project Architecture & Layout

```
Enterprise rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point & lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic-settings configuration management
│   │   └── logging.py       # Application logging configuration
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py      # Async SQLAlchemy PostgreSQL connection
│   │   ├── qdrant.py        # Async Qdrant Vector DB connection
│   │   └── redis.py         # Async Redis connection
│   └── api/
│       ├── __init__.py
│       ├── routes.py        # Main API router setup
│       └── health.py        # Detailed system health check endpoint
├── scripts/
│   ├── verify_imports.py    # Framework import verification script
│   └── verify_connections.py# Live service connectivity verification script
├── tests/
│   ├── __init__.py
│   └── test_health.py       # Pytest suite for API & health check
├── .env                     # Local environment variables
├── .env.example             # Template for environment variables
├── .gitignore               # Ignored files & data folders
├── docker-compose.yml       # Docker Compose setup for PostgreSQL, Qdrant, Redis
├── pyproject.toml           # Project metadata & dependencies
├── requirements.txt         # Pinned core dependencies
└── README.md                # Project setup and execution guide
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.11+ (Python 3.13 recommended)
- **Docker & Docker Compose** (or local PostgreSQL 16, Qdrant 1.11, Redis 7)

---

### 2. Environment Setup

Clone/navigate to project directory and activate virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify environment configuration:
```bash
cp .env.example .env
```

---

### 3. Start Infrastructure Services

#### Option A: Docker Compose (Recommended for Containerized Setup)
```bash
docker compose up -d
```

#### Option B: Local Services (macOS Brew / Local Binaries)
- **PostgreSQL**: `brew services start postgresql@16`
- **Redis**: `brew services start redis`
- **Qdrant**: `/tmp/qdrant --qdrant-config-path ...` or local qdrant binary

---

### 4. Verify Connections, Frameworks & Ingestion Pipeline

Run the verification scripts:

```bash
# Verify LangChain & LangGraph imports
python scripts/verify_imports.py

# Verify database connections (PostgreSQL, Qdrant, Redis)
python scripts/verify_connections.py

# Verify Phase 2 Document Ingestion Pipeline (Parsing, Chunking, Embeddings, Qdrant & PostgreSQL)
python scripts/verify_ingestion.py

# Verify Phase 3 Hybrid Search Pipeline (Vector ANN + Postgres FTS + RRF + Cross-Encoder Rerank)
python scripts/verify_search.py

# Verify Phase 4 Advanced RAG Pipeline (HyDE + CRAG + Self-RAG + LangGraph Workflow)
python scripts/verify_advanced_rag.py

# Verify Phase 5 Text2SQL with Human Approval
python scripts/verify_text2sql.py
```

---

### 5. Run FastAPI Application

Start the development server with Uvicorn:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 6. Core API Endpoints

- **Root Info**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Upload Document**: `POST /api/v1/documents/upload` (Multipart form-data: `file`)
- **List Ingested Documents**: `GET /api/v1/documents`
- **Get Document & Chunks Details**: `GET /api/v1/documents/{document_id}`
- **Hybrid Search**: `POST /api/v1/search` (Body: `{"query": "...", "alpha": 0.7, "top_n": 5}`)
- **Text2SQL Schema**: `GET /api/v1/text2sql/schema`
- **Text2SQL Generate & Stage**: `POST /api/v1/text2sql/generate` (Body: `{"question": "..."}`)
- **Text2SQL Approve & Execute**: `POST /api/v1/text2sql/execute` (Body: `{"query_id": "...", "approved": true}`)
- **Text2SQL Direct Query**: `POST /api/v1/text2sql` (Body: `{"question": "...", "approved": true}`)

---

### 7. Run Unit & Integration Tests

```bash
pytest
```
