import uuid
import pytest
from sqlalchemy import select
from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.models import Document, DocumentChunk, DocumentStatus


@pytest.mark.asyncio
async def test_postgres_document_metadata_crud():
    """Test saving and reading document & chunk metadata records in PostgreSQL."""
    await init_postgres_db()

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    async with AsyncSessionLocal() as session:
        # Create Document record
        doc = Document(
            id=doc_id,
            filename="pg_test.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            total_pages=2,
            total_chunks=1,
            status=DocumentStatus.COMPLETED,
        )
        session.add(doc)

        # Create DocumentChunk record
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            chunk_index=0,
            page_number=1,
            qdrant_point_id=chunk_id,
            character_count=100,
        )
        session.add(chunk)
        await session.commit()

    # Query back
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Document).where(Document.id == doc_id))
        retrieved_doc = res.scalar_one_or_none()

        assert retrieved_doc is not None
        assert retrieved_doc.filename == "pg_test.pdf"
        assert retrieved_doc.status == DocumentStatus.COMPLETED
        assert len(retrieved_doc.chunks) == 1
        assert retrieved_doc.chunks[0].id == chunk_id
