"""Cross-encoder reranker using sentence-transformers CrossEncoder."""
from functools import lru_cache
from typing import List
from sentence_transformers import CrossEncoder
from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult

settings = get_settings()


@lru_cache()
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranker model."""
    logger.info(f"Loading reranker model: {settings.RERANKER_MODEL_NAME}")
    return CrossEncoder(settings.RERANKER_MODEL_NAME)


def rerank(
    query: str,
    candidates: List[ChunkResult],
    top_n: int = None,
) -> List[ChunkResult]:
    """
    Score every candidate with a cross-encoder and return the top-N results.

    The cross-encoder receives (query, chunk_text) pairs and produces a
    relevance logit; higher is more relevant.  Scores are NOT sigmoid-activated
    here so they preserve relative magnitude for downstream inspection.

    Args:
        query:      The original user query string.
        candidates: Pre-merged hybrid results (any length).
        top_n:      Maximum results to return. Defaults to settings.RERANKER_TOP_N.

    Returns:
        Top-N ChunkResult objects sorted by rerank_score descending.
    """
    if top_n is None:
        top_n = settings.RERANKER_TOP_N

    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    raw_scores: List[float] = reranker.predict(pairs).tolist()

    for chunk, score in zip(candidates, raw_scores):
        chunk.rerank_score = round(float(score), 6)

    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    top = ranked[:top_n]

    logger.debug(
        f"Reranker scored {len(candidates)} candidates → returning top {len(top)}. "
        f"Top score: {top[0].rerank_score:.4f}"
    )
    return top
