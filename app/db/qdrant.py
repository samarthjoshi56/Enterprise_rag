import time
from typing import Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


def get_sync_qdrant_client() -> QdrantClient:
    """Return a synchronous Qdrant client instance."""
    url = settings.effective_qdrant_url
    api_key = settings.QDRANT_API_KEY or None
    return QdrantClient(url=url, api_key=api_key, check_compatibility=False)


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return an asynchronous Qdrant client instance."""
    url = settings.effective_qdrant_url
    api_key = settings.QDRANT_API_KEY or None
    return AsyncQdrantClient(url=url, api_key=api_key, check_compatibility=False)


async def init_qdrant_collection() -> None:
    """Ensure the target Qdrant collection exists."""
    from qdrant_client.models import VectorParams, Distance

    client = get_async_qdrant_client()
    try:
        collection_name = settings.QDRANT_COLLECTION_NAME
        exists = await client.collection_exists(collection_name)
        if not exists:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise
    finally:
        await client.close()


async def check_qdrant_connection() -> Dict[str, Any]:
    """Verify Qdrant vector database connectivity and return collection stats."""
    start_time = time.perf_counter()
    client = get_async_qdrant_client()
    try:
        # Get list of collections to verify connectivity
        response = await client.get_collections()
        collections_count = len(response.collections)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "status": "connected",
            "latency_ms": latency_ms,
            "collections_count": collections_count,
            "details": f"Connection successful ({collections_count} collections found)",
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"Qdrant connection check failed: {e}")
        return {
            "status": "error",
            "latency_ms": latency_ms,
            "error": str(e),
        }
    finally:
        await client.close()
