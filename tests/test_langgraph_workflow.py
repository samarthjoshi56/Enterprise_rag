"""Tests for the LangGraph StateGraph workflow in Advanced RAG."""
import pytest
from app.rag.graph import (
    build_advanced_rag_graph,
    AdvancedRAGState,
    route_query_decision,
    route_evaluation_decision,
    GRADE_CORRECT,
    GRADE_INCORRECT,
)
from app.search.models import ChunkResult
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.db.qdrant import init_qdrant_collection


def test_graph_compilation_and_nodes():
    """Graph must compile and contain all required nodes."""
    graph = build_advanced_rag_graph()
    assert graph is not None

    # Check node names in the compiled graph
    node_names = set(graph.nodes.keys())
    expected = {
        "node_query",
        "node_hybrid_retrieve",
        "node_evaluate_relevance",
        "node_correct_retrieval",
        "node_final_context",
    }
    assert expected.issubset(node_names)


def test_routing_query_decision():
    """route_query_decision must route to retrieval when query needs it, or skip to finalize."""
    state_need: AdvancedRAGState = {
        "query": "PostgreSQL indexing",
        "should_retrieve": True,
        "retrieve_token": "[Retrieve]",
        "hypothetical_doc": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retrieval_grade": "",
        "correction_needed": False,
        "correction_count": 0,
        "rewritten_query": None,
        "final_chunks": [],
        "audit_trail": [],
    }
    assert route_query_decision(state_need) == "node_hybrid_retrieve"

    state_no_need: AdvancedRAGState = dict(state_need, should_retrieve=False)
    assert route_query_decision(state_no_need) == "node_final_context"


def test_routing_evaluation_decision():
    """route_evaluation_decision routes to finalize on CORRECT, and correction on INCORRECT."""
    state_correct: AdvancedRAGState = {
        "query": "test",
        "should_retrieve": True,
        "retrieve_token": "",
        "hypothetical_doc": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retrieval_grade": GRADE_CORRECT,
        "correction_needed": False,
        "correction_count": 0,
        "rewritten_query": None,
        "final_chunks": [],
        "audit_trail": [],
    }
    assert route_evaluation_decision(state_correct) == "node_final_context"

    state_incorrect: AdvancedRAGState = dict(
        state_correct, retrieval_grade=GRADE_INCORRECT, correction_count=0
    )
    assert route_evaluation_decision(state_incorrect) == "node_correct_retrieval"

    # When max retries reached, even INCORRECT proceeds to final context
    state_maxed: AdvancedRAGState = dict(
        state_correct, retrieval_grade=GRADE_INCORRECT, correction_count=1
    )
    assert route_evaluation_decision(state_maxed) == "node_final_context"


@pytest.mark.asyncio
async def test_graph_execution_conversational_query():
    """Conversational query should bypass retrieval directly to final context."""
    graph = build_advanced_rag_graph()
    initial_state: AdvancedRAGState = {
        "query": "Hello there, how are you?",
        "should_retrieve": True,
        "retrieve_token": "",
        "hypothetical_doc": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retrieval_grade": "",
        "correction_needed": False,
        "correction_count": 0,
        "rewritten_query": None,
        "final_chunks": [],
        "audit_trail": [],
    }

    result = await graph.ainvoke(initial_state)

    assert result["should_retrieve"] is False
    assert result["retrieve_token"] == "[No Retrieve]"
    assert result["final_chunks"] == []
    # Verify audit trail recorded skip
    assert any("Self-RAG token=[No Retrieve]" in log for log in result["audit_trail"])
