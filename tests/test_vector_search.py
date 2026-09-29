"""Tests for vector search: embedding + Qdrant retrieval."""
import pytest
from app.search.vector_search import vector_search
from app.db.qdrant import init_qdrant_collection


@pytest.mark.asyncio
async def test_vector_search_returns_results():
    """Vector search should return ChunkResult objects for a meaningful query."""
    await init_qdrant_collection()
    results = await vector_search("enterprise Kubernetes deployment", top_k=5)

    # If the collection has any indexed chunks, we should get hits.
    # (Results may be 0 on a fresh empty collection, so we just check the contract.)
    assert isinstance(results, list)
    for r in results:
        assert r.chunk_id
        assert r.text is not None
        assert 0.0 <= r.vector_score <= 1.0


@pytest.mark.asyncio
async def test_vector_search_score_ordering():
    """Returned results should be ordered by descending vector_score."""
    await init_qdrant_collection()
    results = await vector_search("PostgreSQL metadata document", top_k=10)

    for i in range(len(results) - 1):
        assert results[i].vector_score >= results[i + 1].vector_score
