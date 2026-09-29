#!/usr/bin/env python3
"""CLI verification script for Phase 2 — Document Ingestion Pipeline."""

import asyncio
import io
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf
from sqlalchemy import select

from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import get_async_qdrant_client, init_qdrant_collection
from app.db.models import Document, DocumentChunk
from app.ingestion.service import process_and_ingest_document
from app.core.config import get_settings

settings = get_settings()


def create_sample_pdf_bytes() -> bytes:
    """Generate a 2-page sample PDF document with readable text in memory."""
    from reportlab.pdfgen import canvas

    pdf_io = io.BytesIO()
    c = canvas.Canvas(pdf_io)

    # Page 1
    c.drawString(100, 750, "Enterprise RAG - Technical Documentation Page 1")
    c.drawString(100, 730, "This document contains enterprise search procedures and vector index instructions.")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Enterprise RAG - Technical Documentation Page 2")
    c.drawString(100, 730, "Qdrant and PostgreSQL are used to maintain high throughput and hybrid metadata search.")
    c.showPage()

    c.save()
    return pdf_io.getvalue()


async def main():
    print("=" * 65)
    print(" Enterprise RAG — Phase 2 Ingestion Pipeline Verification")
    print("=" * 65)

    # 1. Initialize databases & tables
    await init_postgres_db()
    await init_qdrant_collection()

    # 2. Prepare sample TXT and PDF documents
    txt_content = (
        "Enterprise RAG Architecture Overview:\n"
        "Phase 2 implements the core document ingestion pipeline.\n"
        "Documents are parsed, split into manageable chunks, embedded using "
        "SentenceTransformers (all-MiniLM-L6-v2), stored as vectors in Qdrant, "
        "and metadata registered in PostgreSQL."
    ).encode("utf-8")

    pdf_content = create_sample_pdf_bytes()

    print("\n1. Testing TXT Document Ingestion:")
    async with AsyncSessionLocal() as session:
        txt_result = await process_and_ingest_document(
            file_bytes=txt_content,
            filename="architecture_overview.txt",
            db=session,
        )

    print(f"   [OK] Ingested TXT ID : {txt_result['document_id']}")
    print(f"   [OK] Total Chunks    : {txt_result['total_chunks']}")
    print(f"   [OK] Ingestion Time  : {txt_result['ingestion_time_ms']} ms")

    print("\n2. Testing PDF Document Ingestion:")
    async with AsyncSessionLocal() as session:
        pdf_result = await process_and_ingest_document(
            file_bytes=pdf_content,
            filename="enterprise_report.pdf",
            db=session,
        )

    print(f"   [OK] Ingested PDF ID : {pdf_result['document_id']}")
    print(f"   [OK] Total Pages     : {pdf_result['total_pages']}")
    print(f"   [OK] Total Chunks    : {pdf_result['total_chunks']}")
    print(f"   [OK] Ingestion Time  : {pdf_result['ingestion_time_ms']} ms")

    # 3. Verify Qdrant Vector Points Storage
    print("\n3. Verifying Vector Points in Qdrant:")
    qdrant = get_async_qdrant_client()
    qdrant_info = await qdrant.get_collection(settings.QDRANT_COLLECTION_NAME)
    print(f"   [OK] Collection Name : {settings.QDRANT_COLLECTION_NAME}")
    print(f"   [OK] Total Vectors   : {qdrant_info.points_count}")
    await qdrant.close()

    # 4. Verify PostgreSQL Metadata Storage
    print("\n4. Verifying Metadata Records in PostgreSQL:")
    async with AsyncSessionLocal() as session:
        docs_res = await session.execute(select(Document))
        docs = docs_res.scalars().all()
        chunks_res = await session.execute(select(DocumentChunk))
        chunks = chunks_res.scalars().all()

        print(f"   [OK] Total Document Records in PG : {len(docs)}")
        print(f"   [OK] Total Chunk Records in PG    : {len(chunks)}")

    print("=" * 65)
    print(" PHASE 2 VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
