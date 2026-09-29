"""
Evaluation Runner for RAG retrieval and generation benchmarking.
"""
import time
from typing import List, Dict, Any, Callable, Awaitable, Optional
from app.evaluation.dataset import EvaluationSample, load_evaluation_dataset
from app.evaluation.metrics import (
    compute_retrieval_metrics,
    answer_relevance,
    answer_faithfulness,
)
from app.core.logging import logger


async def run_retrieval_evaluation(
    retriever_func: Callable[[str], Awaitable[List[str]]],
    dataset: Optional[List[EvaluationSample]] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run evaluation benchmark across a dataset using a given retrieval function.

    Args:
        retriever_func: Async callable taking query string and returning list of chunk texts.
        dataset: List of EvaluationSample (loads default if None).
        k: Top-K evaluation cutoff.

    Returns:
        Structured evaluation report with aggregate and per-query scores.
    """
    samples = dataset or load_evaluation_dataset()
    if not samples:
        return {"error": "Evaluation dataset is empty", "sample_count": 0}

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for sample in samples:
        start = time.perf_counter()
        try:
            retrieved_texts = await retriever_func(sample.question)
        except Exception as e:
            logger.warning(f"Retrieval evaluation failed on query '{sample.question}': {e}")
            retrieved_texts = []
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        latencies.append(latency_ms)

        metrics = compute_retrieval_metrics(
            retrieved_texts=retrieved_texts,
            ground_truth_keywords=sample.ground_truth_keywords,
            k=k,
        )

        results.append({
            "id": sample.id,
            "question": sample.question,
            "retrieved_count": len(retrieved_texts),
            "latency_ms": latency_ms,
            "metrics": metrics,
        })

    # Compute aggregate metrics
    num_samples = len(samples)
    mean_precision = sum(r["metrics"][f"precision@{k}"] for r in results) / num_samples
    mean_recall = sum(r["metrics"][f"recall@{k}"] for r in results) / num_samples
    mean_mrr = sum(r["metrics"]["mrr"] for r in results) / num_samples
    mean_hit_rate = sum(r["metrics"][f"hit_rate@{k}"] for r in results) / num_samples
    mean_latency = sum(latencies) / num_samples

    return {
        "sample_count": num_samples,
        "k": k,
        "summary": {
            f"mean_precision@{k}": round(mean_precision, 4),
            f"mean_recall@{k}": round(mean_recall, 4),
            "mean_mrr": round(mean_mrr, 4),
            f"mean_hit_rate@{k}": round(mean_hit_rate, 4),
            "mean_latency_ms": round(mean_latency, 2),
        },
        "query_results": results,
    }


async def run_generation_evaluation(
    rag_func: Callable[[str], Awaitable[Dict[str, Any]]],
    dataset: Optional[List[EvaluationSample]] = None,
) -> Dict[str, Any]:
    """
    Run generation evaluation (answer relevance, faithfulness against context).

    Args:
        rag_func: Async callable returning dict with 'answer' and 'context_chunks'.
    """
    samples = dataset or load_evaluation_dataset()
    results: List[Dict[str, Any]] = []

    for sample in samples:
        try:
            rag_output = await rag_func(sample.question)
            generated_answer = rag_output.get("answer", "")
            context = rag_output.get("context", "")

            rel_score = answer_relevance(sample.question, generated_answer)
            faith_score = answer_faithfulness(context, generated_answer)

            results.append({
                "id": sample.id,
                "question": sample.question,
                "relevance_score": rel_score,
                "faithfulness_score": faith_score,
            })
        except Exception as e:
            logger.warning(f"Generation evaluation failed on query '{sample.question}': {e}")

    num_samples = len(results) or 1
    mean_relevance = sum(r["relevance_score"] for r in results) / num_samples
    mean_faithfulness = sum(r["faithfulness_score"] for r in results) / num_samples

    return {
        "sample_count": len(results),
        "summary": {
            "mean_answer_relevance": round(mean_relevance, 4),
            "mean_faithfulness": round(mean_faithfulness, 4),
        },
        "query_results": results,
    }
