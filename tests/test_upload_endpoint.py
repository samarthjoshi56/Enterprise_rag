import io
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.db.postgres import init_postgres_db
from app.db.qdrant import init_qdrant_collection


@pytest.mark.asyncio
async def test_upload_document_endpoint():
    """Test POST /api/v1/documents/upload with a TXT file."""
    await init_postgres_db()
    await init_qdrant_collection()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        content = b"Enterprise RAG Document Upload Endpoint Integration Test.\nThis is chunk content."
        files = {"file": ("test_ingest.txt", io.BytesIO(content), "text/plain")}

        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 201, response.text
        data = response.json()

        assert "document_id" in data
        assert data["filename"] == "test_ingest.txt"
        assert data["status"] == "COMPLETED"
        assert data["total_chunks"] > 0

        # Query GET /api/v1/documents/{document_id}
        doc_id = data["document_id"]
        get_response = await client.get(f"/api/v1/documents/{doc_id}")
        assert get_response.status_code == 200
        doc_detail = get_response.json()

        assert doc_detail["id"] == doc_id
        assert doc_detail["filename"] == "test_ingest.txt"
        assert doc_detail["status"] == "COMPLETED"
        assert len(doc_detail["chunks"]) > 0
