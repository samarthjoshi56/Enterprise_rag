"""Unit tests for RAG evaluation metrics."""
import pytest
from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    compute_retrieval_metrics,
    answer_relevance,
    answer_faithfulness,
)


def test_precision_at_k():
    """Verify Precision@K calculation."""
    retrieved = [
        "Kubernetes deployment requires running cluster and kubectl manifests.",
        "General notes on computer networking.",
        "Resource limits prevent pod OOM kills.",
        "Cooking pasta requires boiling water.",
        "Kubernetes pods are scheduled by kube-scheduler.",
    ]
    keywords = ["kubernetes", "cluster"]

    # In top 3: chunk 0 has both keywords, chunk 1 doesn't, chunk 2 doesn't have keywords
    # Relevant = 1/3 = 0.3333
    p3 = precision_at_k(retrieved, keywords, k=3)
    assert p3 == 0.3333

    # In top 5: chunk 0 and chunk 4 have "kubernetes" -> 2/5 = 0.4
    p5 = precision_at_k(retrieved, keywords, k=5)
    assert p5 == 0.4


def test_recall_at_k():
    """Verify Recall@K calculation."""
    retrieved = [
        "Kubernetes deployment requires cluster context and kubectl manifests.",
        "Unrelated text.",
    ]
    keywords = ["cluster", "kubectl", "etcd", "monitoring"]

    # Top 2 contains 'cluster' and 'kubectl' -> 2 / 4 = 0.5
    rec = recall_at_k(retrieved, keywords, k=2)
    assert rec == 0.5


def test_reciprocal_rank():
    """Verify MRR / Reciprocal Rank calculation."""
    # First item is relevant -> RR = 1 / 1 = 1.0
    r1 = ["Kubernetes cluster deployment", "other"]
    assert reciprocal_rank(r1, ["kubernetes"]) == 1.0

    # Second item is relevant -> RR = 1 / 2 = 0.5
    r2 = ["other", "Kubernetes cluster deployment"]
    assert reciprocal_rank(r2, ["kubernetes"]) == 0.5

    # Third item is relevant -> RR = 1 / 3 = 0.3333
    r3 = ["other 1", "other 2", "Kubernetes cluster deployment"]
    assert reciprocal_rank(r3, ["kubernetes"]) == 0.3333

    # None relevant -> RR = 0.0
    r_none = ["apple", "orange"]
    assert reciprocal_rank(r_none, ["kubernetes"]) == 0.0


def test_hit_rate_at_k():
    """Verify Hit Rate@K."""
    retrieved = ["other text", "Kubernetes nodes", "more text"]
    assert hit_rate_at_k(retrieved, ["kubernetes"], k=1) == 0.0
    assert hit_rate_at_k(retrieved, ["kubernetes"], k=2) == 1.0


def test_answer_relevance():
    """Verify answer relevance score based on question concept overlap."""
    question = "What are the Kubernetes deployment requirements?"
    relevant_answer = "Kubernetes deployment requirements include a running cluster and kubectl."
    irrelevant_answer = "The capital of France is Paris."

    score_rel = answer_relevance(question, relevant_answer)
    score_irrel = answer_relevance(question, irrelevant_answer)

    assert score_rel > 0.5
    assert score_irrel == 0.0


def test_answer_faithfulness():
    """Verify answer faithfulness / grounding against retrieved context."""
    context = (
        "Enterprise RAG stores document chunks in PostgreSQL for keyword search "
        "and Qdrant for dense vector similarity."
    )
    faithful_answer = "The system uses PostgreSQL for keyword search and Qdrant for vector similarity."
    hallucinated_answer = "The system stores all graphs in Neo4j and streams events to Apache Kafka."

    score_faithful = answer_faithfulness(context, faithful_answer)
    score_hallucinated = answer_faithfulness(context, hallucinated_answer)

    assert score_faithful > 0.5
    assert score_hallucinated < 0.3
