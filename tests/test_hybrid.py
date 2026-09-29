"""Tests for Reciprocal Rank Fusion hybrid merge."""
from app.search.models import ChunkResult
from app.search.hybrid import fuse_results


def _make_chunk(chunk_id: str, vector_score: float = 0.0, keyword_score: float = 0.0) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="test.txt",
        page_number=1,
        chunk_index=0,
        text=f"Text for chunk {chunk_id}",
        vector_score=vector_score,
        keyword_score=keyword_score,
    )


def test_hybrid_deduplication():
    """Chunks present in both vector and keyword results should appear only once."""
    vec = [_make_chunk("a", vector_score=0.9), _make_chunk("b", vector_score=0.7)]
    kw  = [_make_chunk("b", keyword_score=0.8), _make_chunk("c", keyword_score=0.6)]

    merged = fuse_results(vec, kw, alpha=0.7)

    ids = [r.chunk_id for r in merged]
    assert len(ids) == len(set(ids)), "Duplicates found in fused results"
    assert set(ids) == {"a", "b", "c"}


def test_hybrid_score_ordering():
    """Merged results should be sorted by hybrid_score descending."""
    vec = [
        _make_chunk("high", vector_score=0.95),
        _make_chunk("mid", vector_score=0.60),
    ]
    kw  = [_make_chunk("mid", keyword_score=0.90)]

    merged = fuse_results(vec, kw, alpha=0.5)

    for i in range(len(merged) - 1):
        assert merged[i].hybrid_score >= merged[i + 1].hybrid_score


def test_hybrid_pure_vector_alpha_one():
    """With alpha=1.0 keyword scores should not elevate low-vector chunks."""
    vec = [_make_chunk("a", vector_score=0.9), _make_chunk("b", vector_score=0.2)]
    kw  = [_make_chunk("b", keyword_score=1.0)]

    merged = fuse_results(vec, kw, alpha=1.0)
    by_id = {r.chunk_id: r for r in merged}

    # 'a' has much higher vector score; RRF rank bonus for 'b' is small
    assert by_id["a"].hybrid_score >= by_id["b"].hybrid_score


def test_hybrid_empty_inputs():
    """Both empty lists should return an empty merged list."""
    merged = fuse_results([], [], alpha=0.7)
    assert merged == []


def test_hybrid_one_empty_list():
    """When one source is empty the other's results should pass through."""
    vec = [_make_chunk("a", vector_score=0.85)]
    merged = fuse_results(vec, [], alpha=0.7)
    assert len(merged) == 1
    assert merged[0].chunk_id == "a"
