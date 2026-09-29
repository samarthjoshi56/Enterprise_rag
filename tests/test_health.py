import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint returns welcome message."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Enterprise RAG" in data["message"]
        assert data["health"] == "/health"


@pytest.mark.asyncio
async def test_health_endpoint_structure():
    """Test health endpoint returns valid JSON schema and services dict."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        
        assert "status" in data
        assert "services" in data
        assert "postgres" in data["services"]
        assert "qdrant" in data["services"]
        assert "redis" in data["services"]
        assert "frameworks" in data
        assert "langchain" in data["frameworks"]
        assert "langgraph" in data["frameworks"]
