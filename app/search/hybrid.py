"""
Hybrid retrieval: fuse vector and keyword results using Reciprocal Rank Fusion (RRF)
combined with configurable alpha weighting.

RRF formula per candidate:
    rrf_score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)

Final hybrid score:
    hybrid = alpha * normalised_vector_score + (1 - alpha) * normalised_keyword_score
"""
from typing import List, Dict
from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult

settings = get_settings()

_RRF_K = 60  # Standard RRF constant (Cormack et al., 2009)


def fuse_results(
    vector_results: List[ChunkResult],
    keyword_results: List[ChunkResult],
    alpha: float = None,
) -> List[ChunkResult]:
    """
    Merge vector and keyword result lists using Reciprocal Rank Fusion then
    re-score with configurable alpha weighting.

    Args:
        vector_results:  Ordered list from Qdrant (highest vector_score first).
        keyword_results: Ordered list from PostgreSQL FTS (highest keyword_score first).
        alpha:           Weight for vector score in final blend (0–1). Defaults to
                         settings.HYBRID_ALPHA (0.7 = 70% vector, 30% keyword).

    Returns:
        Deduplicated list of ChunkResult, sorted by hybrid_score descending.
    """
    if alpha is None:
        alpha = settings.HYBRID_ALPHA

    # --- Accumulate RRF scores per chunk_id ---
    rrf_scores: Dict[str, float] = {}
    candidates: Dict[str, ChunkResult] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        candidates[cid] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        if cid not in candidates:
            candidates[cid] = chunk
        else:
            # Preserve keyword score on existing entry
            candidates[cid].keyword_score = chunk.keyword_score

    # --- Normalise RRF scores to [0, 1] ---
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

    merged: List[ChunkResult] = []
    for cid, chunk in candidates.items():
        # Weighted combination: alpha * vector + (1 - alpha) * keyword
        hybrid = alpha * chunk.vector_score + (1.0 - alpha) * chunk.keyword_score
        # Blend in the RRF rank as a tie-breaker normalised component
        rrf_norm = rrf_scores[cid] / max_rrf
        chunk.hybrid_score = round(0.8 * hybrid + 0.2 * rrf_norm, 6)
        merged.append(chunk)

    merged.sort(key=lambda c: c.hybrid_score, reverse=True)
    logger.debug(
        f"Fusion: {len(vector_results)} vector + {len(keyword_results)} keyword "
        f"→ {len(merged)} unique candidates"
    )
    return merged
