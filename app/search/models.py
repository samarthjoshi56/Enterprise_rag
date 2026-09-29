"""Shared data models for search requests and responses."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkResult:
    """A single retrieved document chunk with its provenance and scores."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    vector_score: float = 0.0        # Normalised cosine similarity from Qdrant
    keyword_score: float = 0.0       # Normalised BM25-style rank score from PostgreSQL FTS
    hybrid_score: float = 0.0        # RRF / weighted combination of vector + keyword
    rerank_score: Optional[float] = None  # Cross-encoder score after reranking
