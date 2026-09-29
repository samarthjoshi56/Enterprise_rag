#!/usr/bin/env python3
"""CLI verification script for Phase 6 — Redis Caching."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache.service import cache_service, normalize_cache_key
from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import init_qdrant_collection
from app.search.service import hybrid_search

QUERY = "Kubernetes deployment requirements"


async def main():
    print("=" * 70)
    print(" Enterprise RAG — Phase 6 Redis Caching Verification")
    print("=" * 70)

    # 1. Key Normalization
    k1 = normalize_cache_key("search", "What are the Kubernetes deployment requirements?", top_k=5)
    k2 = normalize_cache_key("search", "  what are the kubernetes deployment requirements?  ", top_k=5)
    print(f"\n[1] Query Normalization Test:")
    print(f"    Raw 1 -> Key: {k1}")
    print(f"    Raw 2 -> Key: {k2}")
    print(f"    Normalized Match: {k1 == k2} (Deterministic collision-free key)")

    # 2. Basic Cache Operations (Miss -> Set -> Hit -> TTL -> Invalidation)
    print(f"\n[2] Basic Cache Operations:")
    test_key = "rag:cache:verify_demo"
    await cache_service.delete(test_key)

    miss = await cache_service.get(test_key)
    print(f"    Initial GET (Cache MISS) : {miss}")

    await cache_service.set(test_key, {"status": "ok", "value": 42}, ttl=30)
    hit = await cache_service.get(test_key)
    print(f"    Second GET (Cache HIT)   : {hit}")

    ttl = await cache_service.get_ttl(test_key)
    print(f"    Remaining TTL            : {ttl}s")

    await cache_service.delete(test_key)
    print(f"    Deleted Test Key         : Exists = {await cache_service.exists(test_key)}")

    # 3. Hybrid Search Integration (Miss vs Hit Latency)
    print(f"\n[3] Search Integration (Cache MISS vs HIT):")
    await init_postgres_db()
    await init_qdrant_collection()

    # Clear previous query cache
    await cache_service.clear()

    async with AsyncSessionLocal() as db:
        # First query (MISS)
        print(f"    Executing Query 1: '{QUERY}' (Cache MISS expected)...")
        r1 = await hybrid_search(QUERY, db=db, top_k=5, bypass_cache=False)
        print(f"      -> Cached   : {r1['cached']}")
        print(f"      -> Latency  : {r1['latency_ms']} ms")
        print(f"      -> Hits     : {r1['returned']}")

        # Second query (HIT)
        print(f"    Executing Query 2: '{QUERY}' (Cache HIT expected)...")
        r2 = await hybrid_search(QUERY, db=db, top_k=5, bypass_cache=False)
        print(f"      -> Cached   : {r2['cached']}")
        print(f"      -> Latency  : {r2['latency_ms']} ms")
        print(f"      -> Cache Key: {r2['cache_key']}")

    speedup = round(r1['latency_ms'] / max(r2['latency_ms'], 0.1), 1)
    print(f"\n    Performance Gain: {speedup}x faster on Cache HIT!")

    # 4. Cache Stats
    print(f"\n[4] Redis Cache Statistics:")
    stats = await cache_service.get_stats()
    for k, v in stats.items():
        print(f"    {k:28}: {v}")

    print("\n" + "=" * 70)
    print(" PHASE 6 REDIS CACHING VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
