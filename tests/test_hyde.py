"""Unit and integration tests for HyDE (Hypothetical Document Embeddings)."""
import pytest
from app.rag.hyde import hyde_search, compare_retrieval
from app.rag.llm import generate_hypothetical_document
from app.search.models import ChunkResult
from app.db.qdrant import init_qdrant_collection


@pytest.mark.asyncio
async def test_generate_hypothetical_document():
    """Hypothetical document generation must return a non-empty descriptive passage."""
    query = "What are the production deployment requirements for Kubernetes?"
    doc = await generate_hypothetical_document(query)

    assert isinstance(doc, str)
    assert len(doc) > 30
    assert "Kubernetes" in doc or "deployment" in doc or "technical" in doc.lower()


@pytest.mark.asyncio
async def test_hyde_search_execution():
    """HyDE search should query Qdrant and return (results, hypothetical_document)."""
    await init_qdrant_collection()
    query = "enterprise document chunking architecture"
    results, hypo_doc = await hyde_search(query, top_k=5)

    assert isinstance(results, list)
    assert isinstance(hypo_doc, str)
    assert len(hypo_doc) > 0
    for r in results:
        assert isinstance(r, ChunkResult)
        assert r.chunk_id
        assert 0.0 <= r.vector_score <= 1.0


def test_compare_retrieval_metrics():
    """compare_retrieval must accurately compute Jaccard similarity and disjoint sets."""
    normal = [
        ChunkResult("c1", "d1", "f.txt", 1, 0, "text1"),
        ChunkResult("c2", "d1", "f.txt", 1, 1, "text2"),
    ]
    hyde = [
        ChunkResult("c2", "d1", "f.txt", 1, 1, "text2"),
        ChunkResult("c3", "d1", "f.txt", 1, 2, "text3"),
    ]

    metrics = compare_retrieval(normal, hyde)
    assert metrics["normal_count"] == 2
    assert metrics["hyde_count"] == 2
    assert metrics["overlap_count"] == 1
    assert metrics["common_ids"] == ["c2"]
    assert metrics["normal_only_ids"] == ["c1"]
    assert metrics["hyde_only_ids"] == ["c3"]
    # Jaccard = 1 / 3 = 0.3333
    assert abs(metrics["jaccard_similarity"] - 0.3333) < 0.01
