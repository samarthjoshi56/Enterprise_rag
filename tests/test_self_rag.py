"""Unit tests for Self-RAG reflection tokens and self-assessment mechanisms."""
import pytest
from app.search.models import ChunkResult
from app.rag.self_rag import (
    should_retrieve,
    evaluate_chunk_relevance,
    filter_relevant_chunks,
    decide_more_retrieval_needed,
    validate_answer_support,
    TOKEN_RETRIEVE_YES,
    TOKEN_RETRIEVE_NO,
    TOKEN_IS_REL_YES,
    TOKEN_IS_REL_NO,
    TOKEN_IS_SUP_FULLY,
    TOKEN_IS_SUP_NO,
)


def test_self_rag_should_retrieve_decision():
    """Verify [Retrieve] token triggers correctly for factual queries and skips for chatter."""
    # Factual technical questions need retrieval
    should_get, token = should_retrieve("How do I configure PostgreSQL connection pooling?")
    assert should_get is True
    assert token == TOKEN_RETRIEVE_YES

    # Greetings / chatter do not need retrieval
    should_get, token = should_retrieve("Hello, how are you today?")
    assert should_get is False
    assert token == TOKEN_RETRIEVE_NO

    should_get, token = should_retrieve("thank you very much")
    assert should_get is False
    assert token == TOKEN_RETRIEVE_NO

    # Math expressions do not need retrieval
    should_get, token = should_retrieve("2 + 2")
    assert should_get is False
    assert token == TOKEN_RETRIEVE_NO


def test_self_rag_is_rel_chunk_evaluation():
    """Verify [IsREL] token correctly identifies relevant chunks."""
    query = "PostgreSQL asyncpg connection pool configuration"
    relevant_chunk = ChunkResult(
        chunk_id="c1",
        document_id="d1",
        filename="db.py",
        page_number=1,
        chunk_index=0,
        text="Configure asyncpg connection pool in SQLAlchemy using NullPool or AsyncEngine.",
    )
    irrelevant_chunk = ChunkResult(
        chunk_id="c2",
        document_id="d2",
        filename="recipes.txt",
        page_number=1,
        chunk_index=0,
        text="To make pizza dough, dissolve yeast in warm water with olive oil.",
    )

    is_rel, token, score = evaluate_chunk_relevance(query, relevant_chunk, threshold=-2.0)
    assert is_rel is True
    assert token == TOKEN_IS_REL_YES

    is_rel, token, score = evaluate_chunk_relevance(query, irrelevant_chunk, threshold=-2.0)
    assert is_rel is False
    assert token == TOKEN_IS_REL_NO


def test_self_rag_filter_relevant_chunks():
    """Batch filtering must separate relevant chunks and provide reflection audit."""
    query = "Kubernetes pods scheduling"
    chunks = [
        ChunkResult("c1", "d1", "k8s.txt", 1, 0, "Kubernetes kube-scheduler assigns pods to nodes."),
        ChunkResult("c2", "d2", "other.txt", 1, 0, "The quick brown fox jumps over the lazy dog."),
    ]

    filtered, reflections = filter_relevant_chunks(query, chunks, threshold=-2.0)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c1"
    assert len(reflections) == 2


def test_self_rag_decide_more_retrieval_needed():
    """Decide more retrieval needed based on count of relevant chunks."""
    needed, reason = decide_more_retrieval_needed("any query", [], min_required_chunks=1)
    assert needed is True

    chunk = ChunkResult("c1", "d1", "f.txt", 1, 0, "some relevant text")
    needed, reason = decide_more_retrieval_needed("any query", [chunk], min_required_chunks=1)
    assert needed is False


@pytest.mark.asyncio
async def test_self_rag_validate_answer_support():
    """Verify [IsSUP] hallucination and grounding validation."""
    chunks = [
        ChunkResult(
            chunk_id="c1",
            document_id="d1",
            filename="arch.txt",
            page_number=1,
            chunk_index=0,
            text="Enterprise RAG uses PostgreSQL for relational metadata and Qdrant for vector storage.",
        )
    ]

    # Grounded claim
    grounded_claim = "PostgreSQL stores relational metadata while Qdrant handles vectors."
    is_sup, token = await validate_answer_support(chunks, grounded_claim)
    assert is_sup is True
    assert token == TOKEN_IS_SUP_FULLY

    # Hallucinated / unsupported claim
    hallucinated_claim = "The system uses Cassandra for time-series logs and Neo4j for graphs."
    is_sup, token = await validate_answer_support(chunks, hallucinated_claim)
    assert is_sup is False
    assert token == TOKEN_IS_SUP_NO
