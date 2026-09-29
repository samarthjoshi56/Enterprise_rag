from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.postgres import get_async_db
from app.db.models import Document, DocumentChunk
from app.ingestion.service import process_and_ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf", "txt", "text"}


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Upload a PDF or TXT file to trigger end-to-end ingestion:
    1. Text extraction & metadata parsing
    2. Chunking
    3. Embedding generation
    4. Qdrant vector storage
    5. PostgreSQL metadata persistence
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported formats: .pdf, .txt",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await process_and_ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
            db=db,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@router.get("", summary="List all ingested documents")
async def list_documents(db: AsyncSession = Depends(get_async_db)):
    """List all document metadata records stored in PostgreSQL."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    documents = result.scalars().all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "total_pages": doc.total_pages,
            "total_chunks": doc.total_chunks,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in documents
    ]


@router.get("/{document_id}", summary="Get document metadata and chunks")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Retrieve document status, metadata, and chunk listing by document ID."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = chunks_result.scalars().all()

    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size_bytes": doc.file_size_bytes,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "status": doc.status,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "chunks": [
            {
                "chunk_id": c.id,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "qdrant_point_id": c.qdrant_point_id,
                "character_count": c.character_count,
            }
            for c in chunks
        ],
    }
