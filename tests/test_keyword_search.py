"""Tests for PostgreSQL full-text keyword search."""
import pytest
from app.search.keyword_search import keyword_search
from app.db.postgres import AsyncSessionLocal, init_postgres_db


@pytest.mark.asyncio
async def test_keyword_search_contract():
    """Keyword search should return a list of ChunkResult objects."""
    await init_postgres_db()

    async with AsyncSessionLocal() as session:
        results = await keyword_search(
            "enterprise architecture retrieval", db=session, top_k=10
        )

    assert isinstance(results, list)
    for r in results:
        assert r.chunk_id
        assert r.text is not None
        assert 0.0 <= r.keyword_score <= 1.0


@pytest.mark.asyncio
async def test_keyword_search_score_ordering():
    """Keyword results must be ordered highest score first."""
    await init_postgres_db()

    async with AsyncSessionLocal() as session:
        results = await keyword_search("document vector index", db=session, top_k=10)

    for i in range(len(results) - 1):
        assert results[i].keyword_score >= results[i + 1].keyword_score


@pytest.mark.asyncio
async def test_keyword_search_empty_query_graceful():
    """A query that matches nothing should return an empty list gracefully."""
    await init_postgres_db()

    async with AsyncSessionLocal() as session:
        results = await keyword_search(
            "xyzzy_no_match_token_impossible", db=session, top_k=5
        )

    assert results == []
