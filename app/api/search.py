"""POST /api/v1/search — hybrid search endpoint."""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_async_db
from app.search.service import hybrid_search
from app.search.models import ChunkResult
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User search query")
    vector_top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Max vector hits to retrieve from Qdrant (default: VECTOR_SEARCH_TOP_K)"
    )
    keyword_top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Max keyword hits to retrieve from PostgreSQL (default: KEYWORD_SEARCH_TOP_K)"
    )
    alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Hybrid blend weight: 1.0=pure vector, 0.0=pure keyword (default: HYBRID_ALPHA)"
    )
    top_n: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Number of reranked results to return (default: RERANKER_TOP_N)"
    )
    bypass_cache: bool = Field(
        default=False,
        description="If True, skips Redis cache lookup and forces a fresh search"
    )


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    vector_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: Optional[float]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_candidates: int
    vector_hits: int
    keyword_hits: int
    returned: int
    latency_ms: float
    cached: bool = False
    cache_key: Optional[str] = None


def _chunk_to_item(c: ChunkResult) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        filename=c.filename,
        page_number=c.page_number,
        chunk_index=c.chunk_index,
        text=c.text,
        vector_score=round(c.vector_score, 6),
        keyword_score=round(c.keyword_score, 6),
        hybrid_score=round(c.hybrid_score, 6),
        rerank_score=round(c.rerank_score, 6) if c.rerank_score is not None else None,
    )


@router.post(
    "",
    response_model=SearchResponse,
    summary="Hybrid search: vector + keyword + reranking + Redis caching",
)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Run a hybrid search pipeline:

    1. **Redis Cache** — Check for existing normalized query results
    2. **Vector search** — embed query → Qdrant ANN
    3. **Keyword search** — PostgreSQL full-text search
    4. **Hybrid fusion** — Reciprocal Rank Fusion + alpha weighting
    5. **Reranking** — cross-encoder ms-marco-MiniLM
    6. **Cache Write** — Store results in Redis

    Returns the top-N most relevant document chunks with all intermediate scores.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    try:
        pipeline_result = await hybrid_search(
            query=request.query,
            db=db,
            vector_top_k=request.vector_top_k,
            keyword_top_k=request.keyword_top_k,
            alpha=request.alpha,
            reranker_top_n=request.top_n,
            bypass_cache=request.bypass_cache,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search pipeline failed: {str(e)}",
        )

    return SearchResponse(
        query=pipeline_result["query"],
        results=[_chunk_to_item(c) for c in pipeline_result["results"]],
        total_candidates=pipeline_result["total_candidates"],
        vector_hits=pipeline_result["vector_hits"],
        keyword_hits=pipeline_result["keyword_hits"],
        returned=pipeline_result["returned"],
        latency_ms=pipeline_result["latency_ms"],
        cached=pipeline_result.get("cached", False),
        cache_key=pipeline_result.get("cache_key"),
    )
