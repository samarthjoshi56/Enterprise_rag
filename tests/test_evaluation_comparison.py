"""Integration tests for comparing retrieval configurations."""
import pytest
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.db.qdrant import init_qdrant_collection
from app.evaluation.comparison import compare_retrieval_configurations
from app.evaluation.dataset import load_evaluation_dataset


@pytest.mark.asyncio
async def test_compare_retrieval_configurations():
    """Verify multi-configuration benchmark executes and produces comparative table."""
    await init_postgres_db()
    await init_qdrant_collection()

    samples = load_evaluation_dataset()[:2]  # run on first 2 samples for fast testing

    async with AsyncSessionLocal() as db:
        comparison = await compare_retrieval_configurations(
            db=db,
            dataset=samples,
            k=3,
        )

    assert "configurations" in comparison
    configs = comparison["configurations"]

    assert "Vector-Only" in configs
    assert "Keyword-Only" in configs
    assert "Hybrid" in configs
    assert "Hybrid + Reranking" in configs
    assert "Advanced RAG (HyDE+CRAG)" in configs

    for name, metrics in configs.items():
        assert "mean_precision@3" in metrics
        assert "mean_recall@3" in metrics
        assert "mean_mrr" in metrics
        assert "mean_hit_rate@3" in metrics
        assert "mean_latency_ms" in metrics

    assert "best_configurations" in comparison
