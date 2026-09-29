#!/usr/bin/env python3
"""CLI verification script for Phase 3 — Hybrid Search Pipeline."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import init_qdrant_collection
from app.search.service import hybrid_search
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search


QUERY = "Kubernetes deployment requirements"


async def main():
    print("=" * 65)
    print(" Enterprise RAG — Phase 3 Hybrid Search Pipeline Verification")
    print("=" * 65)
    print(f"\n  Query: \"{QUERY}\"\n")

    await init_postgres_db()
    await init_qdrant_collection()

    # 1. Vector Search
    print("1. Vector Search (Qdrant ANN):")
    vec_results = await vector_search(QUERY, top_k=5)
    if vec_results:
        for r in vec_results[:3]:
            print(f"   [OK] score={r.vector_score:.4f} | {r.filename} p{r.page_number} | {r.text[:80]!r}")
    else:
        print("   [INFO] No vector hits (collection may need documents ingested first)")

    # 2. Keyword Search
    print("\n2. Keyword Search (PostgreSQL FTS):")
    async with AsyncSessionLocal() as db:
        kw_results = await keyword_search(QUERY, db=db, top_k=5)
    if kw_results:
        for r in kw_results[:3]:
            print(f"   [OK] score={r.keyword_score:.4f} | {r.filename} p{r.page_number} | {r.text[:80]!r}")
    else:
        print("   [INFO] No keyword hits")

    # 3. Full Hybrid Pipeline
    print("\n3. Full Hybrid Pipeline (Fusion → Rerank):")
    async with AsyncSessionLocal() as db:
        result = await hybrid_search(QUERY, db=db)

    print(f"   Vector hits    : {result['vector_hits']}")
    print(f"   Keyword hits   : {result['keyword_hits']}")
    print(f"   Fused candidates: {result['total_candidates']}")
    print(f"   Returned after rerank: {result['returned']}")
    print(f"   Latency        : {result['latency_ms']} ms")

    print("\n   Top Reranked Results:")
    for i, r in enumerate(result["results"], 1):
        print(
            f"   #{i}  rerank={r.rerank_score:.4f} | hybrid={r.hybrid_score:.4f} | "
            f"{r.filename} p{r.page_number}"
        )
        print(f"        {r.text[:100]!r}")

    print("\n" + "=" * 65)
    print(" PHASE 3 HYBRID SEARCH VERIFICATION COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
