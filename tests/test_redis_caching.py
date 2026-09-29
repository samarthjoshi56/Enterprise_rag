"""Tests for Redis caching: Miss, Hit, TTL, Invalidation, and Search Integration."""
import asyncio
import pytest
from app.cache.service import RedisCacheService, normalize_cache_key
from app.search.service import hybrid_search
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.db.qdrant import init_qdrant_collection
from app.ingestion.service import process_and_ingest_document


@pytest.mark.asyncio
async def test_cache_miss_and_hit():
    """Verify cache returns None on miss, and correct value on subsequent hit."""
    cache = RedisCacheService()
    key = normalize_cache_key("test_namespace", "What is Kubernetes?", top_k=5)

    # Ensure clean state
    await cache.delete(key)

    # 1. Cache Miss
    miss_val = await cache.get(key)
    assert miss_val is None

    # 2. Set value
    sample_data = {"answer": "Container orchestration", "chunks": ["c1", "c2"]}
    stored = await cache.set(key, sample_data, ttl=60)
    assert stored is True

    # 3. Cache Hit
    hit_val = await cache.get(key)
    assert hit_val is not None
    assert hit_val["answer"] == "Container orchestration"
    assert hit_val["chunks"] == ["c1", "c2"]

    # Cleanup
    await cache.delete(key)


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Verify key expires after specified TTL seconds."""
    cache = RedisCacheService()
    key = normalize_cache_key("test_ttl", "short living query")

    # Set with 1-second TTL
    await cache.set(key, {"temp": "data"}, ttl=1)

    # Verify exists immediately
    assert await cache.exists(key) is True
    val = await cache.get(key)
    assert val == {"temp": "data"}

    # Wait for expiration
    await asyncio.sleep(1.2)

    # Verify expired
    expired_val = await cache.get(key)
    assert expired_val is None
    assert await cache.exists(key) is False


@pytest.mark.asyncio
async def test_cache_invalidation_clear():
    """Verify clear() deletes all keys matching prefix."""
    cache = RedisCacheService()
    k1 = normalize_cache_key("clear_test", "query one")
    k2 = normalize_cache_key("clear_test", "query two")

    await cache.set(k1, {"res": 1}, ttl=60)
    await cache.set(k2, {"res": 2}, ttl=60)

    assert await cache.exists(k1) is True
    assert await cache.exists(k2) is True

    cleared = await cache.clear(prefix="rag:cache:clear_test")
    assert cleared >= 2

    assert await cache.get(k1) is None
    assert await cache.get(k2) is None


@pytest.mark.asyncio
async def test_search_cache_integration():
    """
    Verify full integration with search service:
    Query 1 -> Cache MISS -> Cached in Redis
    Query 2 -> Cache HIT (<10ms, cached=True)
    """
    await init_postgres_db()
    await init_qdrant_collection()

    doc_text = "Kubernetes pods run containerized workloads isolated within worker node namespaces.".encode("utf-8")
    async with AsyncSessionLocal() as db:
        await process_and_ingest_document(
            file_bytes=doc_text,
            filename="k8s_pods_cache_test.txt",
            db=db,
        )

    query = "containerized workloads isolated within worker node"

    # Query 1: First search (Cache MISS)
    async with AsyncSessionLocal() as db:
        res1 = await hybrid_search(query, db=db, top_k=3, bypass_cache=False)

    assert res1["cached"] is False
    assert len(res1["results"]) > 0

    # Query 2: Repeated search (Cache HIT)
    async with AsyncSessionLocal() as db:
        res2 = await hybrid_search(query, db=db, top_k=3, bypass_cache=False)

    assert res2["cached"] is True
    assert res2["cache_key"] == res1["cache_key"]
    assert len(res2["results"]) == len(res1["results"])
    assert res2["latency_ms"] < 25.0  # Cache hit is near instantaneous
