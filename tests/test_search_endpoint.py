"""Integration tests for POST /api/v1/search endpoint."""
import io
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.db.qdrant import init_qdrant_collection
from app.ingestion.service import process_and_ingest_document


@pytest.fixture(scope="module")
async def ingest_test_document():
    """Ingest a single test document so search has data to work with."""
    await init_postgres_db()
    await init_qdrant_collection()

    content = (
        "Kubernetes deployment requirements: You need a running cluster, "
        "a container registry, and a configured kubectl context. "
        "Deployments are described using YAML manifests and applied with kubectl apply. "
        "Resource limits and health probes are mandatory for production workloads. "
        "Use namespaces to isolate environments such as staging and production. "
        "Enterprise RAG uses PostgreSQL for metadata and Qdrant for vector storage."
    ).encode("utf-8")

    async with AsyncSessionLocal() as db:
        result = await process_and_ingest_document(
            file_bytes=content,
            filename="k8s_deployment_guide.txt",
            db=db,
        )
    return result


@pytest.mark.asyncio
async def test_search_endpoint_returns_200(ingest_test_document):
    """Search endpoint must return 200 with valid results."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/search",
            json={"query": "Kubernetes deployment requirements"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert "query" in data
    assert data["query"] == "Kubernetes deployment requirements"


@pytest.mark.asyncio
async def test_search_response_schema(ingest_test_document):
    """Every result item must carry all required fields and score types."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/search",
            json={"query": "enterprise vector PostgreSQL"},
        )

    assert response.status_code == 200
    data = response.json()

    for item in data["results"]:
        assert "chunk_id" in item
        assert "document_id" in item
        assert "filename" in item
        assert "text" in item
        assert "vector_score" in item
        assert "keyword_score" in item
        assert "hybrid_score" in item
        assert "rerank_score" in item


@pytest.mark.asyncio
async def test_search_top_n_respected(ingest_test_document):
    """top_n parameter must cap the number of returned results."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/search",
            json={"query": "Kubernetes cluster namespace", "top_n": 2},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 2


@pytest.mark.asyncio
async def test_search_empty_query_rejected():
    """Empty query string must return HTTP 422 (validation error)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/search", json={"query": ""})

    assert response.status_code == 422
