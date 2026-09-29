"""
Search orchestration service.

Full pipeline:
    query → [vector search ‖ keyword search] → fuse (RRF + alpha) → rerank → top-N results
"""
import asyncio
import time
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search
from app.search.hybrid import fuse_results
from app.search.reranker import rerank
from app.search.models import ChunkResult
from app.cache.service import cache_service, normalize_cache_key

settings = get_settings()


def _chunk_to_dict(c: ChunkResult) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "document_id": c.document_id,
        "filename": c.filename,
        "page_number": c.page_number,
        "chunk_index": c.chunk_index,
        "text": c.text,
        "vector_score": c.vector_score,
        "keyword_score": c.keyword_score,
        "hybrid_score": c.hybrid_score,
        "rerank_score": c.rerank_score,
    }


def _dict_to_chunk(d: dict) -> ChunkResult:
    return ChunkResult(
        chunk_id=d["chunk_id"],
        document_id=d["document_id"],
        filename=d["filename"],
        page_number=d.get("page_number"),
        chunk_index=d.get("chunk_index", 0),
        text=d["text"],
        vector_score=d.get("vector_score", 0.0),
        keyword_score=d.get("keyword_score", 0.0),
        hybrid_score=d.get("hybrid_score", 0.0),
        rerank_score=d.get("rerank_score"),
    )


async def hybrid_search(
    query: str,
    db: AsyncSession,
    vector_top_k: int = None,
    keyword_top_k: int = None,
    alpha: float = None,
    reranker_top_n: int = None,
    top_k: int = None,
    bypass_cache: bool = False,
) -> dict:
    """
    Run the full hybrid search + reranking pipeline with Redis caching.

    Check Redis -> Cache HIT (return cached result in <5ms)
                 -> Cache MISS (run vector + keyword + rerank, store in Redis)
    """
    start = time.perf_counter()

    vector_k = vector_top_k or top_k or settings.VECTOR_SEARCH_TOP_K
    keyword_k = keyword_top_k or top_k or settings.KEYWORD_SEARCH_TOP_K
    _alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
    top_n = reranker_top_n or top_k or settings.RERANKER_TOP_N

    cache_key = normalize_cache_key(
        "hybrid_search",
        query,
        vk=vector_k,
        kk=keyword_k,
        alpha=_alpha,
        tn=top_n,
    )

    # 1. Check Redis Cache
    if not bypass_cache and settings.REDIS_CACHE_ENABLED:
        cached_payload = await cache_service.get(cache_key)
        if cached_payload is not None:
            hit_latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(f"Hybrid search CACHE HIT for query='{query[:40]}' ({hit_latency_ms}ms)")
            return {
                "query": query,
                "results": [_dict_to_chunk(item) for item in cached_payload["results"]],
                "total_candidates": cached_payload["total_candidates"],
                "vector_hits": cached_payload["vector_hits"],
                "keyword_hits": cached_payload["keyword_hits"],
                "returned": cached_payload["returned"],
                "latency_ms": hit_latency_ms,
                "cached": True,
                "cache_key": cache_key,
            }

    # 2. Cache MISS: Run vector and keyword search concurrently
    vector_task = asyncio.create_task(vector_search(query, top_k=vector_k))
    keyword_task = asyncio.create_task(keyword_search(query, db=db, top_k=keyword_k))
    vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

    # 3. Fuse results
    candidates: List[ChunkResult] = fuse_results(
        vector_results, keyword_results, alpha=_alpha
    )

    total_candidates = len(candidates)

    # 4. Rerank
    final_results: List[ChunkResult] = rerank(query, candidates, top_n=top_n)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    response_dict = {
        "query": query,
        "results": final_results,
        "total_candidates": total_candidates,
        "vector_hits": len(vector_results),
        "keyword_hits": len(keyword_results),
        "returned": len(final_results),
        "latency_ms": latency_ms,
        "cached": False,
        "cache_key": cache_key,
    }

    # 5. Store in Redis Cache
    if settings.REDIS_CACHE_ENABLED:
        cacheable_payload = {
            "results": [_chunk_to_dict(c) for c in final_results],
            "total_candidates": total_candidates,
            "vector_hits": len(vector_results),
            "keyword_hits": len(keyword_results),
            "returned": len(final_results),
        }
        await cache_service.set(cache_key, cacheable_payload, ttl=settings.REDIS_CACHE_TTL)

    logger.info(
        f"Hybrid search | query='{query[:60]}' | "
        f"vector={len(vector_results)} keyword={len(keyword_results)} "
        f"fused={total_candidates} reranked={len(final_results)} "
        f"latency={latency_ms}ms [cached=False]"
    )

    return response_dict
