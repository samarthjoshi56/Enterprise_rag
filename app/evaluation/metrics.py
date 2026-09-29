"""
Evaluation metrics for RAG retrieval and generation.

Retrieval Metrics:
- Precision@K: Fraction of top-K chunks that contain relevant ground-truth information.
- Recall@K: Fraction of expected ground-truth keywords/concepts captured in top-K chunks.
- Reciprocal Rank (RR): 1 / rank of first relevant chunk.
- Hit Rate@K: 1.0 if at least one relevant chunk appears in top-K, 0.0 otherwise.

Generation Metrics:
- Answer Relevance: Semantic and lexical alignment between query and generated answer.
- Answer Faithfulness / Grounding: Degree to which statements in the answer are supported by retrieved context.
"""
import re
from typing import List, Set, Dict, Any


def is_chunk_relevant(
    chunk_text: str,
    ground_truth_keywords: List[str],
    min_keyword_matches: int = 1,
) -> bool:
    """Determine whether a chunk is relevant based on ground-truth keyword presence."""
    if not chunk_text or not ground_truth_keywords:
        return False
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in ground_truth_keywords if kw.lower() in text_lower)
    return matches >= min_keyword_matches


def precision_at_k(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K:
    Number of relevant chunks among the top K retrieved chunks divided by K.
    """
    if k <= 0 or not retrieved_texts:
        return 0.0

    top_k = retrieved_texts[:k]
    relevant_count = sum(
        1 for t in top_k if is_chunk_relevant(t, ground_truth_keywords)
    )
    return round(relevant_count / len(top_k), 4)


def recall_at_k(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K:
    Fraction of ground-truth concepts/keywords found within the top-K chunks.
    """
    if not ground_truth_keywords or not retrieved_texts:
        return 0.0

    top_k_combined = " ".join(retrieved_texts[:k]).lower()
    matched = sum(1 for kw in ground_truth_keywords if kw.lower() in top_k_combined)
    return round(matched / len(ground_truth_keywords), 4)


def reciprocal_rank(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
) -> float:
    """
    Calculate Reciprocal Rank (RR):
    1 / rank of the first relevant chunk in the retrieved list (1-indexed).
    Returns 0.0 if no relevant chunk is retrieved.
    """
    for rank, text in enumerate(retrieved_texts, start=1):
        if is_chunk_relevant(text, ground_truth_keywords):
            return round(1.0 / rank, 4)
    return 0.0


def hit_rate_at_k(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Calculate Hit Rate@K:
    1.0 if at least one chunk in the top K is relevant, 0.0 otherwise.
    """
    for text in retrieved_texts[:k]:
        if is_chunk_relevant(text, ground_truth_keywords):
            return 1.0
    return 0.0


def compute_retrieval_metrics(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Compute all standard retrieval metrics for a single query."""
    return {
        f"precision@{k}": precision_at_k(retrieved_texts, ground_truth_keywords, k=k),
        f"recall@{k}": recall_at_k(retrieved_texts, ground_truth_keywords, k=k),
        "mrr": reciprocal_rank(retrieved_texts, ground_truth_keywords),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_texts, ground_truth_keywords, k=k),
    }


def answer_relevance(question: str, answer: str) -> float:
    """
    Evaluate answer relevance score [0.0 - 1.0] by measuring core question
    concept overlap in the answer.
    """
    if not question.strip() or not answer.strip():
        return 0.0

    stop_words = {
        "what", "is", "are", "how", "to", "the", "a", "an", "in", "on", "for",
        "of", "do", "does", "can", "you", "tell", "me", "about", "please", "why"
    }
    q_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", question.lower())) - stop_words
    if not q_tokens:
        return 1.0

    a_lower = answer.lower()
    matches = sum(1 for t in q_tokens if t in a_lower)
    return round(matches / len(q_tokens), 4)


def answer_faithfulness(context: str, answer: str) -> float:
    """
    Evaluate answer faithfulness / grounding [0.0 - 1.0].
    Measures the proportion of key informative terms in the answer supported by the context.
    """
    if not answer.strip():
        return 1.0
    if not context.strip():
        return 0.0

    ans_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{4,}\b", answer.lower()))
    if not ans_tokens:
        return 1.0

    ctx_lower = context.lower()
    supported = sum(1 for t in ans_tokens if t in ctx_lower)
    return round(supported / len(ans_tokens), 4)
