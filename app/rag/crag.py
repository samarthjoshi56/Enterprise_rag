"""
CRAG: Corrective Retrieval-Augmented Generation (Yan et al., 2024).

Components:
1. Retrieval Evaluator: Grades relevance of retrieved chunks to query.
   Categorizes retrieval confidence into:
   - CORRECT: Retrieved context is high-quality and directly relevant.
   - AMBIGUOUS: Context has marginal relevance; needs supplementary retrieval.
   - INCORRECT: Context is irrelevant or missing; requires query reformulation & re-retrieval.
2. Corrective Action:
   - Rewrites the query to focus on core semantic concepts.
   - Triggers corrective retrieval with rewritten query + HyDE.
   - Filters out noisy/irrelevant chunks (Knowledge Refinement).
"""
from typing import List, Tuple, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult
from app.search.reranker import get_reranker
from app.search.hybrid import fuse_results
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search
from app.rag.llm import rewrite_query
from app.rag.hyde import hyde_search

settings = get_settings()

# Quality states
GRADE_CORRECT = "CORRECT"
GRADE_AMBIGUOUS = "AMBIGUOUS"
GRADE_INCORRECT = "INCORRECT"


def evaluate_retrieval_quality(
    query: str,
    chunks: List[ChunkResult],
    correct_threshold: float = None,
    incorrect_threshold: float = None,
) -> Tuple[str, List[ChunkResult]]:
    """
    Grade retrieval quality using neural cross-encoder relevance scores.

    Classification logic:
    - If no chunks: INCORRECT
    - If top score >= correct_threshold: CORRECT (filter chunks with score >= incorrect_threshold)
    - If top score < incorrect_threshold: INCORRECT
    - Otherwise: AMBIGUOUS (marginal relevance)

    Returns:
        (grade, filtered_chunks)
    """
    if correct_threshold is None:
        correct_threshold = settings.CRAG_RELEVANCE_THRESHOLD  # default e.g. -2.0 for ms-marco
    if incorrect_threshold is None:
        incorrect_threshold = correct_threshold - 4.0

    if not chunks:
        logger.info("CRAG Evaluator: No chunks retrieved -> Grade: INCORRECT")
        return GRADE_INCORRECT, []

    # Score candidates using the cross-encoder
    reranker = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs).tolist()

    for chunk, s in zip(chunks, scores):
        chunk.rerank_score = round(float(s), 6)

    # Sort chunks by score descending
    sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    top_score = sorted_chunks[0].rerank_score

    logger.debug(
        f"CRAG Evaluator: top_score={top_score:.4f}, correct_thresh={correct_threshold}, "
        f"incorrect_thresh={incorrect_threshold}"
    )

    if top_score >= correct_threshold:
        grade = GRADE_CORRECT
        # Knowledge Refinement: retain only chunks that meet minimum acceptable score
        filtered = [c for c in sorted_chunks if c.rerank_score >= incorrect_threshold]
    elif top_score < incorrect_threshold:
        grade = GRADE_INCORRECT
        filtered = []
    else:
        grade = GRADE_AMBIGUOUS
        filtered = sorted_chunks

    logger.info(f"CRAG Evaluator: Grade={grade} (retained {len(filtered)}/{len(chunks)} chunks)")
    return grade, filtered


async def execute_corrective_retrieval(
    query: str,
    db: AsyncSession,
    reason: str = "low retrieval quality",
    top_k: int = 10,
) -> Tuple[List[ChunkResult], str]:
    """
    Perform corrective retrieval:
    1. Reformulate / rewrite query to eliminate ambiguity.
    2. Execute parallel vector search + keyword search on rewritten query.
    3. Execute HyDE search on rewritten query.
    4. Fuse all candidates together.

    Returns:
        (improved_candidates, rewritten_query)
    """
    rewritten = await rewrite_query(query, reason=reason)
    logger.info(f"CRAG Corrective Retrieval: Rewritten query='{rewritten}'")

    # Retrieve using rewritten query
    vec_results = await vector_search(rewritten, top_k=top_k)
    kw_results = await keyword_search(rewritten, db=db, top_k=top_k)
    hyde_results, _ = await hyde_search(rewritten, top_k=top_k)

    # Fuse normal results first
    base_fused = fuse_results(vec_results, kw_results, alpha=0.5)

    # Fuse in HyDE candidates
    all_fused = fuse_results(base_fused, hyde_results, alpha=0.6)

    logger.info(
        f"CRAG Corrective Retrieval produced {len(all_fused)} candidate chunks "
        f"(vector={len(vec_results)}, keyword={len(kw_results)}, hyde={len(hyde_results)})"
    )
    return all_fused, rewritten
