"""Unit tests for CRAG (Corrective Retrieval-Augmented Generation)."""
import pytest
from app.search.models import ChunkResult
from app.rag.crag import (
    evaluate_retrieval_quality,
    execute_corrective_retrieval,
    GRADE_CORRECT,
    GRADE_INCORRECT,
    GRADE_AMBIGUOUS,
)
from app.rag.llm import rewrite_query
from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import init_qdrant_collection


def test_crag_evaluator_correct_grade():
    """Directly relevant chunks should receive GRADE_CORRECT."""
    query = "Kubernetes deployment requirements"
    chunks = [
        ChunkResult(
            chunk_id="c1",
            document_id="d1",
            filename="k8s.txt",
            page_number=1,
            chunk_index=0,
            text="Kubernetes deployment requirements include a running cluster, kubectl, and yaml manifests.",
        ),
        ChunkResult(
            chunk_id="c2",
            document_id="d1",
            filename="k8s.txt",
            page_number=1,
            chunk_index=1,
            text="General notes on cloud computing.",
        ),
    ]

    grade, filtered = evaluate_retrieval_quality(
        query=query,
        chunks=chunks,
        correct_threshold=-2.0,
    )
    assert grade == GRADE_CORRECT
    assert len(filtered) >= 1
    assert filtered[0].chunk_id == "c1"


def test_crag_evaluator_incorrect_grade():
    """Unrelated chunks should receive GRADE_INCORRECT."""
    query = "Kubernetes deployment requirements"
    chunks = [
        ChunkResult(
            chunk_id="c1",
            document_id="d1",
            filename="cooking.txt",
            page_number=1,
            chunk_index=0,
            text="Baking a chocolate cake requires flour, sugar, and cocoa powder.",
        )
    ]

    grade, filtered = evaluate_retrieval_quality(
        query=query,
        chunks=chunks,
        correct_threshold=5.0,
        incorrect_threshold=0.0,
    )
    assert grade == GRADE_INCORRECT
    assert filtered == []


def test_crag_evaluator_empty_chunks():
    """Empty chunk list should immediately be classified as INCORRECT."""
    grade, filtered = evaluate_retrieval_quality(
        query="Any query",
        chunks=[],
    )
    assert grade == GRADE_INCORRECT
    assert filtered == []


@pytest.mark.asyncio
async def test_crag_query_rewriter():
    """Query rewriter should strip noise and produce cleaner search terms."""
    noisy_query = "Can you please tell me what is the Kubernetes deployment requirements?"
    rewritten = await rewrite_query(noisy_query, reason="poor retrieval")

    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    # Common conversational fillers should be stripped
    assert "please" not in rewritten.lower()
    assert "tell" not in rewritten.lower()


@pytest.mark.asyncio
async def test_crag_corrective_retrieval_execution():
    """Corrective retrieval should query databases and return fused candidates."""
    await init_postgres_db()
    await init_qdrant_collection()

    async with AsyncSessionLocal() as db:
        candidates, rewritten = await execute_corrective_retrieval(
            query="What are the Kubernetes deployment steps?",
            db=db,
            reason="ambiguous retrieval",
            top_k=5,
        )

    assert isinstance(candidates, list)
    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
