"""
Evaluation Dataset Loader.

Loads benchmark question-answer evaluation pairs from JSON storage.
"""
import json
import os
from typing import List, Optional
from dataclasses import dataclass
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


@dataclass
class EvaluationSample:
    """A single evaluation benchmark test case."""
    id: str
    question: str
    ground_truth_keywords: List[str]
    expected_answer: str
    target_service: str = "kubernetes"


_DEFAULT_BENCHMARK = [
    EvaluationSample(
        id="eval-001",
        question="What are the Kubernetes deployment requirements?",
        ground_truth_keywords=["cluster", "kubectl", "container registry", "yaml", "manifest"],
        expected_answer="Kubernetes deployment requirements include a running cluster, container registry, kubectl context, and YAML manifests.",
        target_service="kubernetes",
    ),
    EvaluationSample(
        id="eval-002",
        question="How do you achieve high availability in enterprise Kubernetes worker nodes?",
        ground_truth_keywords=["availability zones", "three", "etcd", "worker nodes", "persistent ssd"],
        expected_answer="Deploy worker nodes across at least three availability zones with persistent SSD storage for etcd clusters.",
        target_service="kubernetes",
    ),
    EvaluationSample(
        id="eval-003",
        question="How do you prevent OOM kills on Kubernetes pods?",
        ground_truth_keywords=["resource requests", "limits", "cpu", "memory", "oom"],
        expected_answer="Production workloads must specify CPU and memory resource requests and limits to prevent OOM kills.",
        target_service="kubernetes",
    ),
    EvaluationSample(
        id="eval-004",
        question="What database systems are used for metadata and vector storage in Enterprise RAG?",
        ground_truth_keywords=["postgresql", "qdrant", "metadata", "vector"],
        expected_answer="Enterprise RAG uses PostgreSQL for relational metadata and Qdrant for dense vector storage.",
        target_service="architecture",
    ),
]


def load_evaluation_dataset(filepath: Optional[str] = None) -> List[EvaluationSample]:
    """
    Load benchmark questions from JSON file. Falls back to default benchmark if file is missing.
    """
    target_path = filepath or settings.EVALUATION_DATASET_PATH

    if not os.path.exists(target_path):
        logger.warning(f"Evaluation dataset file '{target_path}' not found. Using default benchmarks.")
        return _DEFAULT_BENCHMARK

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[EvaluationSample] = []
        for item in data:
            samples.append(
                EvaluationSample(
                    id=item.get("id", "eval-unknown"),
                    question=item["question"],
                    ground_truth_keywords=item.get("ground_truth_keywords", []),
                    expected_answer=item.get("expected_answer", ""),
                    target_service=item.get("target_service", "general"),
                )
            )
        logger.info(f"Loaded {len(samples)} evaluation benchmark samples from '{target_path}'")
        return samples
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}. Using defaults.")
        return _DEFAULT_BENCHMARK
