"""Unit tests for the evaluation runner."""
import pytest
from app.evaluation.runner import run_retrieval_evaluation, run_generation_evaluation
from app.evaluation.dataset import EvaluationSample


@pytest.mark.asyncio
async def test_run_retrieval_evaluation_mock():
    """Verify evaluation runner aggregates metrics accurately."""
    sample = EvaluationSample(
        id="test-1",
        question="What are Kubernetes deployment requirements?",
        ground_truth_keywords=["cluster", "kubectl"],
        expected_answer="Cluster and kubectl.",
    )

    async def mock_retriever(q: str):
        return [
            "Kubernetes deployment requires a running cluster.",
            "kubectl context must be configured.",
            "Unrelated information.",
        ]

    report = await run_retrieval_evaluation(
        retriever_func=mock_retriever,
        dataset=[sample],
        k=3,
    )

    assert report["sample_count"] == 1
    summary = report["summary"]
    # Chunk 0 has 'cluster', chunk 1 has 'kubectl' -> 2/3 relevant
    assert summary["mean_precision@3"] > 0.6
    assert summary["mean_recall@3"] == 1.0  # both keywords matched
    assert summary["mean_mrr"] == 1.0       # first hit is relevant
    assert summary["mean_hit_rate@3"] == 1.0


@pytest.mark.asyncio
async def test_run_generation_evaluation_mock():
    """Verify generation evaluation runner calculates relevance and faithfulness."""
    sample = EvaluationSample(
        id="test-1",
        question="What are Kubernetes deployment requirements?",
        ground_truth_keywords=["cluster", "kubectl"],
        expected_answer="Cluster and kubectl.",
    )

    async def mock_rag(q: str):
        return {
            "answer": "Kubernetes deployment requirements include a running cluster and kubectl.",
            "context": "Kubernetes deployment requires a running cluster and kubectl manifests.",
        }

    report = await run_generation_evaluation(rag_func=mock_rag, dataset=[sample])

    assert report["sample_count"] == 1
    summary = report["summary"]
    assert summary["mean_answer_relevance"] > 0.5
    assert summary["mean_faithfulness"] > 0.5
