import time
from typing import Dict, Any
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


def get_redis_client() -> aioredis.Redis:
    """Return an async Redis client."""
    return aioredis.from_url(
        settings.effective_redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


async def check_redis_connection() -> Dict[str, Any]:
    """Verify Redis connectivity and return server info."""
    start_time = time.perf_counter()
    client = get_redis_client()
    try:
        ping_result = await client.ping()
        info = await client.info("server")
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        redis_version = info.get("redis_version", "unknown")
        
        return {
            "status": "connected" if ping_result else "degraded",
            "latency_ms": latency_ms,
            "version": redis_version,
            "details": "PONG received",
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"Redis connection check failed: {e}")
        return {
            "status": "error",
            "latency_ms": latency_ms,
            "error": str(e),
        }
    finally:
        await client.aclose()
