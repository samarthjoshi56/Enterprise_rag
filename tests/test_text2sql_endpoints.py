"""API endpoint tests for Text2SQL with Human Approval."""
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.db.postgres import init_postgres_db


@pytest.mark.asyncio
async def test_get_text2sql_schema_endpoint():
    """GET /api/v1/text2sql/schema should return allowed tables and column definitions."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/text2sql/schema")

    assert response.status_code == 200
    data = response.json()
    assert "allowed_tables" in data
    assert "incidents" in data["allowed_tables"]
    assert "documents" in data["allowed_tables"]
    assert "schema" in data


@pytest.mark.asyncio
async def test_text2sql_two_step_approval_endpoints():
    """Test full two-step HTTP workflow: POST /generate -> POST /execute."""
    await init_postgres_db()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Step 1: Generate & stage
        gen_resp = await client.post(
            "/api/v1/text2sql/generate",
            json={"question": "What are the top 5 Kubernetes incidents?"},
        )
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()
        assert gen_data["status"] == "PENDING_APPROVAL"
        assert gen_data["is_safe"] is True
        query_id = gen_data["query_id"]

        # Step 2: Approve and execute
        exec_resp = await client.post(
            "/api/v1/text2sql/execute",
            json={"query_id": query_id, "approved": True},
        )
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "APPROVED"
        assert exec_data["row_count"] > 0
        assert len(exec_data["rows"]) > 0


@pytest.mark.asyncio
async def test_text2sql_endpoint_rejection():
    """Test rejection via POST /execute."""
    await init_postgres_db()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        gen_resp = await client.post(
            "/api/v1/text2sql/generate",
            json={"question": "How many documents are stored?"},
        )
        query_id = gen_resp.json()["query_id"]

        exec_resp = await client.post(
            "/api/v1/text2sql/execute",
            json={"query_id": query_id, "approved": False},
        )
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "REJECTED"
        assert exec_data["row_count"] == 0


@pytest.mark.asyncio
async def test_text2sql_direct_endpoint_with_approval():
    """Test convenience direct endpoint with approved=True."""
    await init_postgres_db()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/text2sql",
            json={
                "question": "What are the top 5 Kubernetes incidents?",
                "approved": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_safe"] is True
        assert "execution" in data
        assert data["execution"]["status"] == "APPROVED"
