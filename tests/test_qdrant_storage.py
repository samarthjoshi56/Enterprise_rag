import uuid
import pytest
from qdrant_client.models import PointStruct
from app.db.qdrant import get_async_qdrant_client, init_qdrant_collection
from app.core.config import get_settings

settings = get_settings()


@pytest.mark.asyncio
async def test_qdrant_upsert_and_retrieve():
    """Test initializing collection and upserting vector point into Qdrant."""
    await init_qdrant_collection()
    client = get_async_qdrant_client()

    point_id = str(uuid.uuid4())
    fake_vector = [0.1] * settings.EMBEDDING_DIMENSION

    # Upsert test point
    upsert_res = await client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=fake_vector,
                payload={
                    "chunk_id": point_id,
                    "document_id": "test-doc-id",
                    "filename": "qdrant_test.txt",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text": "Qdrant insertion test payload",
                    "character_count": 29,
                },
            )
        ],
    )
    assert upsert_res.status == "completed"

    # Retrieve point by ID
    points = await client.retrieve(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        ids=[point_id],
    )
    assert len(points) == 1
    assert points[0].payload["filename"] == "qdrant_test.txt"
    assert points[0].payload["text"] == "Qdrant insertion test payload"

    await client.close()
