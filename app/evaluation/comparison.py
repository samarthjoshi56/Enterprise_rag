"""
Configuration Comparison Module.

Evaluates and compares different retrieval strategies across standard benchmarks:
1. Vector-Only (Qdrant ANN)
2. Keyword-Only (PostgreSQL Full-Text Search)
3. Hybrid Search (RRF fused Vector + Keyword, unranked)
4. Hybrid + Reranking (Phase 3: Fused + Cross-Encoder)
5. Advanced RAG (Phase 4: HyDE + CRAG + Reranking)
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.dataset import EvaluationSample, load_evaluation_dataset
from app.evaluation.runner import run_retrieval_evaluation
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search
from app.search.hybrid import fuse_results
from app.search.reranker import rerank
from app.rag.service import run_advanced_rag
from app.core.logging import logger


async def compare_retrieval_configurations(
    db: AsyncSession,
    dataset: Optional[List[EvaluationSample]] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run the benchmark suite across all 5 retrieval configurations
    and produce a comparative evaluation table.
    """
    samples = dataset or load_evaluation_dataset()
    logger.info(f"Starting retrieval configuration comparison across {len(samples)} benchmark queries")

    # 1. Vector-Only Retriever
    async def retrieve_vector_only(query: str) -> List[str]:
        hits = await vector_search(query, top_k=k)
        return [h.text for h in hits]

    # 2. Keyword-Only Retriever
    async def retrieve_keyword_only(query: str) -> List[str]:
        hits = await keyword_search(query, db=db, top_k=k)
        return [h.text for h in hits]

    # 3. Hybrid (Unranked) Retriever
    async def retrieve_hybrid_unranked(query: str) -> List[str]:
        vec = await vector_search(query, top_k=k)
        kw = await keyword_search(query, db=db, top_k=k)
        fused = fuse_results(vec, kw, alpha=0.7)
        return [h.text for h in fused[:k]]

    # 4. Hybrid + Cross-Encoder Reranking
    async def retrieve_hybrid_reranked(query: str) -> List[str]:
        vec = await vector_search(query, top_k=10)
        kw = await keyword_search(query, db=db, top_k=10)
        fused = fuse_results(vec, kw, alpha=0.7)
        ranked = rerank(query, fused, top_n=k)
        return [h.text for h in ranked]

    # 5. Advanced RAG (HyDE + CRAG + Rerank)
    async def retrieve_advanced_rag(query: str) -> List[str]:
        out = await run_advanced_rag(query, db=db, top_k=10, top_n=k)
        return [h.text for h in out.get("final_chunks", [])]

    configs = {
        "Vector-Only": retrieve_vector_only,
        "Keyword-Only": retrieve_keyword_only,
        "Hybrid": retrieve_hybrid_unranked,
        "Hybrid + Reranking": retrieve_hybrid_reranked,
        "Advanced RAG (HyDE+CRAG)": retrieve_advanced_rag,
    }

    comparison_results: Dict[str, Any] = {}

    for name, retriever in configs.items():
        logger.info(f"Evaluating configuration: '{name}'...")
        eval_run = await run_retrieval_evaluation(
            retriever_func=retriever,
            dataset=samples,
            k=k,
        )
        comparison_results[name] = eval_run["summary"]

    # Identify best performing configuration per metric
    best_mrr_config = max(comparison_results.items(), key=lambda x: x[1]["mean_mrr"])[0]
    best_recall_config = max(comparison_results.items(), key=lambda x: x[1][f"mean_recall@{k}"])[0]

    return {
        "benchmark_sample_count": len(samples),
        "k": k,
        "configurations": comparison_results,
        "best_configurations": {
            "highest_mrr": best_mrr_config,
            f"highest_recall@{k}": best_recall_config,
        },
    }
