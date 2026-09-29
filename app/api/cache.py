"""
FastAPI endpoints for Redis Cache monitoring and management.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.cache.service import cache_service

router = APIRouter(prefix="/cache", tags=["Cache"])


class CacheClearRequest(BaseModel):
    prefix: Optional[str] = Field(
        default=None,
        description="Optional key prefix pattern to clear. Defaults to all RAG cache keys.",
    )


class CacheClearResponse(BaseModel):
    cleared_keys_count: int
    message: str


@router.get(
    "/stats",
    summary="Get Cache Health and Key Metrics",
    description="Returns current Redis caching statistics, keyspace hits/misses, and cached RAG key counts.",
)
async def get_cache_stats():
    """Retrieve cache operational metrics."""
    return await cache_service.get_stats()


@router.post(
    "/clear",
    response_model=CacheClearResponse,
    summary="Invalidate / Clear Cached Queries",
    description="Flushes cached search and RAG results from Redis.",
)
async def clear_cache(request: CacheClearRequest = CacheClearRequest()):
    """Clear cached search and RAG queries."""
    count = await cache_service.clear(prefix=request.prefix)
    return CacheClearResponse(
        cleared_keys_count=count,
        message=f"Successfully cleared {count} cached entries from Redis.",
    )


@router.delete(
    "/keys/{key:path}",
    summary="Delete Specific Cache Key",
    description="Removes a specific cache key from Redis.",
)
async def delete_cache_key(key: str):
    """Delete a specific key from cache."""
    deleted = await cache_service.delete(key)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{key}' not found or could not be deleted.",
        )
    return {"key": key, "deleted": True}
