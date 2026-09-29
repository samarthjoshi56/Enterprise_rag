"""Unit tests for Evaluation dataset loading."""
import pytest
from app.evaluation.dataset import load_evaluation_dataset, EvaluationSample


def test_load_evaluation_dataset():
    """Verify evaluation dataset loads with all required benchmark fields."""
    samples = load_evaluation_dataset()

    assert len(samples) >= 4
    for s in samples:
        assert isinstance(s, EvaluationSample)
        assert s.id.startswith("eval-")
        assert len(s.question) > 10
        assert len(s.ground_truth_keywords) > 0
        assert len(s.expected_answer) > 10
        assert s.target_service in ("kubernetes", "architecture", "general")
