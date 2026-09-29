"""
Redis Cache Service for Enterprise RAG.

Provides:
- Get, Set, Delete, Invalidate / Clear with configurable TTL
- Normalized cache key generation for queries and parameters
- Resilient fallback if Redis is temporarily unreachable
- Cache hit/miss/memory operational metrics
"""
import json
import hashlib
from typing import Any, Optional, Dict, List
from app.core.config import get_settings
from app.core.logging import logger
from app.db.redis import get_redis_client

settings = get_settings()


def normalize_cache_key(namespace: str, query: str, **params: Any) -> str:
    """
    Generate a deterministic, collision-resistant cache key.
    Normalizes query by lowercasing, stripping extra whitespace,
    and sorting optional parameters.
    """
    normalized_q = " ".join(query.strip().lower().split())

    # Build sorted string of auxiliary parameters
    param_str = ""
    if params:
        sorted_items = sorted((k, str(v)) for k, v in params.items() if v is not None)
        param_str = "|" + "|".join(f"{k}={v}" for k, v in sorted_items)

    raw_payload = f"{normalized_q}{param_str}"
    digest = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:16]
    prefix = settings.REDIS_CACHE_PREFIX.rstrip(":")
    return f"{prefix}:{namespace}:{digest}"


class RedisCacheService:
    """Async Redis cache client with robust error handling and JSON serialization."""

    def __init__(self, default_ttl: Optional[int] = None):
        self.default_ttl = default_ttl or settings.REDIS_CACHE_TTL
        self.enabled = settings.REDIS_CACHE_ENABLED

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a deserialized JSON value from cache. Returns None on miss/error."""
        if not self.enabled:
            return None

        client = get_redis_client()
        try:
            val = await client.get(key)
            if val is not None:
                logger.debug(f"Redis Cache HIT: key='{key}'")
                return json.loads(val)
            logger.debug(f"Redis Cache MISS: key='{key}'")
            return None
        except Exception as e:
            logger.warning(f"Redis GET failed for key '{key}': {e}")
            return None
        finally:
            await client.aclose()

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a JSON-serializable value in cache with TTL."""
        if not self.enabled:
            return False

        effective_ttl = ttl if ttl is not None else self.default_ttl
        client = get_redis_client()
        try:
            serialized = json.dumps(value, default=str)
            if effective_ttl and effective_ttl > 0:
                await client.set(key, serialized, ex=effective_ttl)
            else:
                await client.set(key, serialized)
            logger.debug(f"Redis Cache SET: key='{key}', ttl={effective_ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for key '{key}': {e}")
            return False
        finally:
            await client.aclose()

    async def delete(self, key: str) -> bool:
        """Delete a single key from cache."""
        client = get_redis_client()
        try:
            deleted_count = await client.delete(key)
            return deleted_count > 0
        except Exception as e:
            logger.warning(f"Redis DELETE failed for key '{key}': {e}")
            return False
        finally:
            await client.aclose()

    async def exists(self, key: str) -> bool:
        """Check if a cache key exists."""
        if not self.enabled:
            return False
        client = get_redis_client()
        try:
            return bool(await client.exists(key))
        except Exception as e:
            logger.warning(f"Redis EXISTS failed for key '{key}': {e}")
            return False
        finally:
            await client.aclose()

    async def get_ttl(self, key: str) -> int:
        """Return the remaining TTL in seconds for a key (-2 if not exists, -1 if no TTL)."""
        client = get_redis_client()
        try:
            return await client.ttl(key)
        except Exception as e:
            logger.warning(f"Redis TTL failed for key '{key}': {e}")
            return -2
        finally:
            await client.aclose()

    async def clear(self, prefix: Optional[str] = None) -> int:
        """
        Invalidate all keys matching the prefix.
        Defaults to settings.REDIS_CACHE_PREFIX.
        """
        match_pattern = f"{prefix or settings.REDIS_CACHE_PREFIX}*"
        client = get_redis_client()
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=match_pattern, count=100)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"Redis Cache Invalidation: cleared {deleted} keys matching '{match_pattern}'")
            return deleted
        except Exception as e:
            logger.warning(f"Redis CLEAR failed for pattern '{match_pattern}': {e}")
            return 0
        finally:
            await client.aclose()

    async def get_stats(self) -> Dict[str, Any]:
        """Return operational cache statistics."""
        client = get_redis_client()
        try:
            info = await client.info("stats")
            keyspace = await client.info("keyspace")
            prefix = settings.REDIS_CACHE_PREFIX
            keys = await client.keys(f"{prefix}*")

            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = round(hits / total, 4) if total > 0 else 0.0

            return {
                "enabled": self.enabled,
                "default_ttl_seconds": self.default_ttl,
                "cached_rag_keys_count": len(keys),
                "total_redis_keyspace_hits": hits,
                "total_redis_keyspace_misses": misses,
                "keyspace_hit_rate": hit_rate,
            }
        except Exception as e:
            logger.warning(f"Redis get_stats failed: {e}")
            return {
                "enabled": self.enabled,
                "default_ttl_seconds": self.default_ttl,
                "error": str(e),
            }
        finally:
            await client.aclose()


# Singleton service instance
cache_service = RedisCacheService()
