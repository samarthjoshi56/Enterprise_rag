import time
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.core.logging import logger
from app.db.models import Document, DocumentChunk, DocumentStatus
from app.db.qdrant import get_async_qdrant_client, init_qdrant_collection
from app.ingestion.loader import parse_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embeddings import generate_embeddings_batch

settings = get_settings()


async def process_and_ingest_document(
    file_bytes: bytes,
    filename: str,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    Complete end-to-end document ingestion pipeline:
    1. Parse document (PDF/TXT)
    2. Save initial PENDING metadata record in PostgreSQL
    3. Split into text chunks
    4. Generate vector embeddings for all chunks
    5. Store points (vector + payload) in Qdrant
    6. Save chunk records and update status to COMPLETED in PostgreSQL
    """
    start_time = time.perf_counter()

    # 1. Parse Document
    parsed_doc = parse_document(file_bytes, filename)
    doc_id = parsed_doc.document_id

    # 2. Record Document in PostgreSQL as PROCESSING
    db_doc = Document(
        id=doc_id,
        filename=filename,
        file_type=parsed_doc.file_type,
        file_size_bytes=parsed_doc.file_size_bytes,
        total_pages=parsed_doc.total_pages,
        total_chunks=0,
        status=DocumentStatus.PROCESSING,
    )
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)

    try:
        # 3. Chunk Document
        chunks = chunk_document(parsed_doc)
        if not chunks:
            raise ValueError(f"Document '{filename}' produced no text chunks.")

        # 4. Generate Embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = generate_embeddings_batch(chunk_texts)

        # 5. Store Vectors in Qdrant
        await init_qdrant_collection()
        qdrant_client = get_async_qdrant_client()

        qdrant_points = [
            PointStruct(
                id=c.chunk_id,
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "character_count": c.character_count,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]

        await qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=qdrant_points,
        )
        await qdrant_client.close()

        # 6. Store Chunk Metadata & Update Status in PostgreSQL
        db_chunks = [
            DocumentChunk(
                id=c.chunk_id,
                document_id=doc_id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                qdrant_point_id=c.chunk_id,
                character_count=c.character_count,
                chunk_text=c.text,
            )
            for c in chunks
        ]
        db.add_all(db_chunks)

        db_doc.total_chunks = len(chunks)
        db_doc.status = DocumentStatus.COMPLETED
        await db.commit()

        # Invalidate search cache so new document content is immediately searchable
        try:
            from app.cache.service import cache_service
            await cache_service.clear()
        except Exception as e:
            logger.warning(f"Cache invalidation after ingestion failed: {e}")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"Successfully ingested '{filename}' (ID: {doc_id}): "
            f"{len(chunks)} chunks embedded and stored in {elapsed_ms} ms."
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "file_type": parsed_doc.file_type,
            "file_size_bytes": parsed_doc.file_size_bytes,
            "total_pages": parsed_doc.total_pages,
            "total_chunks": len(chunks),
            "status": DocumentStatus.COMPLETED,
            "ingestion_time_ms": elapsed_ms,
        }

    except Exception as e:
        logger.error(f"Failed to ingest document '{filename}': {e}")
        db_doc.status = DocumentStatus.FAILED
        db_doc.error_message = str(e)
        await db.commit()
        raise
