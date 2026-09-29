"""Tests for the cross-encoder reranker."""
from app.search.models import ChunkResult
from app.search.reranker import rerank


def _make_chunk(chunk_id: str, text: str, hybrid_score: float = 0.5) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="test.pdf",
        page_number=1,
        chunk_index=0,
        text=text,
        hybrid_score=hybrid_score,
    )


def test_reranker_populates_rerank_score():
    """Every returned result must have a rerank_score set."""
    query = "enterprise document search"
    chunks = [
        _make_chunk("a", "Enterprise search is built on Qdrant and PostgreSQL."),
        _make_chunk("b", "Python is a general-purpose programming language."),
        _make_chunk("c", "Document retrieval uses vector embeddings for similarity."),
    ]
    reranked = rerank(query, chunks, top_n=3)

    assert len(reranked) == 3
    for r in reranked:
        assert r.rerank_score is not None


def test_reranker_returns_at_most_top_n():
    """Reranker must respect the top_n limit."""
    query = "Kubernetes deployment requirements"
    chunks = [_make_chunk(f"c{i}", f"Chunk content number {i}") for i in range(10)]
    reranked = rerank(query, chunks, top_n=3)

    assert len(reranked) <= 3


def test_reranker_score_ordering():
    """Results must be ordered by rerank_score descending."""
    query = "vector store similarity"
    chunks = [
        _make_chunk("a", "Qdrant stores dense vectors for semantic similarity search."),
        _make_chunk("b", "Redis is an in-memory key-value store."),
        _make_chunk("c", "Semantic search relies on cosine distance in embedding space."),
    ]
    reranked = rerank(query, chunks, top_n=3)

    for i in range(len(reranked) - 1):
        assert reranked[i].rerank_score >= reranked[i + 1].rerank_score


def test_reranker_empty_candidates():
    """Empty input should return empty list without error."""
    result = rerank("some query", [], top_n=5)
    assert result == []
