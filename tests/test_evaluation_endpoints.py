"""FastAPI endpoint tests for Evaluation framework."""
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.db.postgres import init_postgres_db
from app.db.qdrant import init_qdrant_collection


@pytest.mark.asyncio
async def test_get_evaluation_dataset_endpoint():
    """GET /api/v1/evaluation/dataset should return benchmark questions."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/evaluation/dataset")

    assert response.status_code == 200
    data = response.json()
    assert "total_samples" in data
    assert data["total_samples"] >= 4
    assert len(data["samples"]) > 0


@pytest.mark.asyncio
async def test_run_evaluation_endpoint():
    """POST /api/v1/evaluation/run should execute evaluation on active pipeline."""
    await init_postgres_db()
    await init_qdrant_collection()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/evaluation/run", json={"k": 3})

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "mean_mrr" in data["summary"]
    assert "query_results" in data


@pytest.mark.asyncio
async def test_compare_evaluation_endpoint():
    """POST /api/v1/evaluation/compare should return side-by-side configuration metrics."""
    await init_postgres_db()
    await init_qdrant_collection()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/evaluation/compare", json={"k": 3})

    assert response.status_code == 200
    data = response.json()
    assert "configurations" in data
    assert "Hybrid + Reranking" in data["configurations"]
    assert "best_configurations" in data
