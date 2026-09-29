#!/usr/bin/env python3
"""CLI verification script for Phase 7 — Evaluation Framework and Comparative Benchmarks."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import init_qdrant_collection
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.metrics import (
    compute_retrieval_metrics,
    answer_relevance,
    answer_faithfulness,
)
from app.evaluation.comparison import compare_retrieval_configurations


async def main():
    print("=" * 75)
    print(" Enterprise RAG — Phase 7 Evaluation Framework & Benchmarks")
    print("=" * 75)

    await init_postgres_db()
    await init_qdrant_collection()

    # 1. Dataset Loading
    samples = load_evaluation_dataset()
    print(f"\n[1] Benchmark Dataset Loaded: {len(samples)} benchmark test questions")
    for s in samples[:3]:
        print(f"    - [{s.id}] \"{s.question}\" (Keywords: {s.ground_truth_keywords[:3]}...)")

    # 2. Metric Computation Demonstration
    print(f"\n[2] Metric Calculations Demo:")
    demo_chunks = [
        "Kubernetes deployment requirements: You need a running cluster and kubectl manifests.",
        "Production workloads should specify resource limits and health probes.",
        "Unrelated documentation about database migrations.",
    ]
    demo_kw = ["cluster", "kubectl", "manifest"]
    metrics = compute_retrieval_metrics(demo_chunks, demo_kw, k=3)
    print(f"    Sample Retrieved Chunks: {len(demo_chunks)} items")
    print(f"    Precision@3 : {metrics['precision@3']}")
    print(f"    Recall@3    : {metrics['recall@3']}")
    print(f"    MRR         : {metrics['mrr']}")
    print(f"    Hit Rate@3  : {metrics['hit_rate@3']}")

    # Generation metrics demo
    q = "What are the Kubernetes deployment requirements?"
    faithful_ans = "Deploying on Kubernetes requires a cluster and kubectl manifests with resource limits."
    hallucinated_ans = "Kubernetes runs directly inside a WebAssembly sandbox on Cloudflare workers."
    context = " ".join(demo_chunks)

    print(f"\n    Generation Metrics Demo:")
    print(f"    Faithful Answer   -> Relevance: {answer_relevance(q, faithful_ans):.2f}, Faithfulness: {answer_faithfulness(context, faithful_ans):.2f}")
    print(f"    Hallucinated Ans  -> Relevance: {answer_relevance(q, hallucinated_ans):.2f}, Faithfulness: {answer_faithfulness(context, hallucinated_ans):.2f}")

    # 3. Multi-Configuration Comparative Benchmark
    print(f"\n[3] Running Comparative Benchmark Across 5 Retrieval Configurations (k=3)...")
    async with AsyncSessionLocal() as db:
        comparison = await compare_retrieval_configurations(db=db, dataset=samples[:3], k=3)

    print("\n" + "-" * 75)
    print(f"{'Retrieval Configuration':<28} | {'P@3':<7} | {'R@3':<7} | {'MRR':<7} | {'Hit@3':<7} | {'Latency':<9}")
    print("-" * 75)

    for name, m in comparison["configurations"].items():
        print(
            f"{name:<28} | "
            f"{m['mean_precision@3']:<7.3f} | "
            f"{m['mean_recall@3']:<7.3f} | "
            f"{m['mean_mrr']:<7.3f} | "
            f"{m['mean_hit_rate@3']:<7.3f} | "
            f"{m['mean_latency_ms']:<6.1f} ms"
        )
    print("-" * 75)

    print(f"\n    Summary Findings:")
    print(f"    - Highest MRR      : {comparison['best_configurations']['highest_mrr']}")
    print(f"    - Highest Recall@3 : {comparison['best_configurations']['highest_recall@3']}")

    print("\n" + "=" * 75)
    print(" PHASE 7 EVALUATION BENCHMARK VERIFICATION COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
