"""
LangGraph workflow for Advanced RAG (HyDE, CRAG, Self-RAG).

Architecture:
               [START]
                  │
                  ▼
            [node_query]  (Self-RAG [Retrieve] decision)
                  │
          ┌───────┴───────┐
          │ (Need Retrieval)
          ▼
   [node_hybrid_retrieve] (Hybrid Search + HyDE retrieval)
          │
          ▼
 [node_evaluate_relevance] (CRAG Grader + Self-RAG [IsREL])
          │
          ├──────────────────────────┐
          │ (Grade != CORRECT        │ (Grade == CORRECT
          │  and retries < max)      │  or max retries reached)
          ▼                          │
[node_correct_retrieval] (CRAG)      │
          │                          │
          └───────────┬──────────────┘
                      │
                      ▼
            [node_final_context] (Cross-Encoder Rerank & Final Context)
                      │
                      ▼
                    [END]
"""
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search
from app.search.hybrid import fuse_results
from app.search.reranker import rerank
from app.rag.hyde import hyde_search
from app.rag.crag import (
    evaluate_retrieval_quality,
    execute_corrective_retrieval,
    GRADE_CORRECT,
    GRADE_AMBIGUOUS,
    GRADE_INCORRECT,
)
from app.rag.self_rag import (
    should_retrieve,
    filter_relevant_chunks,
    TOKEN_RETRIEVE_YES,
    TOKEN_RETRIEVE_NO,
)

settings = get_settings()


class AdvancedRAGState(TypedDict):
    """State schema for the Advanced RAG graph."""
    query: str
    should_retrieve: bool
    retrieve_token: str
    hypothetical_doc: Optional[str]
    retrieved_chunks: List[ChunkResult]
    relevant_chunks: List[ChunkResult]
    retrieval_grade: str
    correction_needed: bool
    correction_count: int
    rewritten_query: Optional[str]
    final_chunks: List[ChunkResult]
    audit_trail: List[str]


# =====================================================================
# Node Implementations
# =====================================================================

async def node_query(state: AdvancedRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze query and decide if external retrieval is required (Self-RAG [Retrieve])."""
    query = state["query"]
    needs_retrieval, token = should_retrieve(query)

    trail = list(state.get("audit_trail", []))
    trail.append(f"Query Node: Analyzed query. Self-RAG token={token}")

    return {
        "should_retrieve": needs_retrieval,
        "retrieve_token": token,
        "correction_count": state.get("correction_count", 0),
        "audit_trail": trail,
    }


async def node_hybrid_retrieve(state: AdvancedRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute hybrid retrieval (Vector ANN + Postgres FTS) and HyDE search."""
    query = state["query"]
    db = config.get("configurable", {}).get("db")
    top_k = config.get("configurable", {}).get("top_k", settings.VECTOR_SEARCH_TOP_K)

    tasks = [
        vector_search(query, top_k=top_k),
        keyword_search(query, db=db, top_k=top_k) if db else asyncio.sleep(0, result=[]),
    ]

    hypo_doc = None
    if settings.HYDE_ENABLED:
        tasks.append(hyde_search(query, top_k=top_k))

    results = await asyncio.gather(*tasks)
    vec_results = results[0]
    kw_results = results[1]

    # Combine vector and keyword
    fused = fuse_results(vec_results, kw_results, alpha=settings.HYBRID_ALPHA)

    # Blend HyDE if enabled
    if settings.HYDE_ENABLED and len(results) > 2:
        hyde_results, hypo_doc = results[2]
        fused = fuse_results(fused, hyde_results, alpha=0.6)

    trail = list(state.get("audit_trail", []))
    trail.append(
        f"Retrieval Node: Retrieved {len(fused)} candidates "
        f"(vector={len(vec_results)}, keyword={len(kw_results)}, "
        f"hyde={len(results[2][0]) if settings.HYDE_ENABLED and len(results) > 2 else 0})"
    )

    return {
        "retrieved_chunks": fused,
        "hypothetical_doc": hypo_doc,
        "audit_trail": trail,
    }


async def node_evaluate_relevance(state: AdvancedRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Evaluate retrieval quality:
    - CRAG retrieval evaluator (CORRECT / AMBIGUOUS / INCORRECT)
    - Self-RAG [IsREL] token per chunk filtering
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    threshold = config.get("configurable", {}).get(
        "relevance_threshold", settings.CRAG_RELEVANCE_THRESHOLD
    )

    # 1. CRAG evaluation
    grade, refined_chunks = evaluate_retrieval_quality(
        query=query,
        chunks=chunks,
        correct_threshold=threshold,
    )

    # 2. Self-RAG [IsREL] filter
    relevant_chunks, reflections = filter_relevant_chunks(
        query=query,
        chunks=refined_chunks,
        threshold=threshold,
    )

    correction_needed = (grade in (GRADE_INCORRECT, GRADE_AMBIGUOUS))

    trail = list(state.get("audit_trail", []))
    trail.append(
        f"Relevance Evaluation Node: CRAG Grade={grade} | "
        f"Self-RAG relevant chunks={len(relevant_chunks)}/{len(chunks)} | "
        f"Correction needed={correction_needed}"
    )

    return {
        "retrieval_grade": grade,
        "relevant_chunks": relevant_chunks,
        "correction_needed": correction_needed,
        "audit_trail": trail,
    }


async def node_correct_retrieval(state: AdvancedRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """CRAG Corrective Retrieval: Rewrite query and execute fallback retrieval."""
    query = state["query"]
    db = config.get("configurable", {}).get("db")
    top_k = config.get("configurable", {}).get("top_k", settings.VECTOR_SEARCH_TOP_K)
    count = state.get("correction_count", 0) + 1

    improved_chunks, rewritten_query = await execute_corrective_retrieval(
        query=query,
        db=db,
        reason=f"CRAG Grade {state.get('retrieval_grade', 'POOR')}",
        top_k=top_k,
    )

    trail = list(state.get("audit_trail", []))
    trail.append(
        f"Correction Node (attempt #{count}): Rewrote query to '{rewritten_query}' | "
        f"Retrieved {len(improved_chunks)} improved candidates"
    )

    return {
        "retrieved_chunks": improved_chunks,
        "rewritten_query": rewritten_query,
        "correction_count": count,
        "audit_trail": trail,
    }


async def node_final_context(state: AdvancedRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Rerank candidates using the cross-encoder and finalize the best context."""
    query = state["query"]
    top_n = config.get("configurable", {}).get("top_n", settings.RERANKER_TOP_N)

    # Prefer relevant chunks; fallback to retrieved chunks
    candidates = state.get("relevant_chunks") or state.get("retrieved_chunks", [])

    final_chunks = rerank(query=query, candidates=candidates, top_n=top_n)

    trail = list(state.get("audit_trail", []))
    trail.append(
        f"Final Context Node: Cross-Encoder reranked {len(candidates)} candidates -> "
        f"Top {len(final_chunks)} final context chunks selected"
    )

    return {
        "final_chunks": final_chunks,
        "audit_trail": trail,
    }


# =====================================================================
# Conditional Edge Routers
# =====================================================================

def route_query_decision(state: AdvancedRAGState) -> str:
    """Decide whether to execute retrieval or jump straight to final context."""
    if state.get("should_retrieve", True):
        return "node_hybrid_retrieve"
    return "node_final_context"


def route_evaluation_decision(state: AdvancedRAGState) -> str:
    """
    Decide whether context is good or requires corrective retrieval:
    - If CORRECT -> continue to final context
    - If INCORRECT / AMBIGUOUS and retries remain -> correct retrieval
    - If max retries exceeded -> continue to final context
    """
    grade = state.get("retrieval_grade", GRADE_CORRECT)
    count = state.get("correction_count", 0)
    max_retries = settings.CRAG_MAX_REWRITES

    if grade == GRADE_CORRECT or count >= max_retries:
        return "node_final_context"
    return "node_correct_retrieval"


# =====================================================================
# Graph Construction & Compilation
# =====================================================================

def build_advanced_rag_graph():
    """Build and compile the Advanced RAG StateGraph."""
    graph = StateGraph(AdvancedRAGState)

    # Add Nodes
    graph.add_node("node_query", node_query)
    graph.add_node("node_hybrid_retrieve", node_hybrid_retrieve)
    graph.add_node("node_evaluate_relevance", node_evaluate_relevance)
    graph.add_node("node_correct_retrieval", node_correct_retrieval)
    graph.add_node("node_final_context", node_final_context)

    # Add Edges
    graph.add_edge(START, "node_query")

    graph.add_conditional_edges(
        "node_query",
        route_query_decision,
        {
            "node_hybrid_retrieve": "node_hybrid_retrieve",
            "node_final_context": "node_final_context",
        },
    )

    graph.add_edge("node_hybrid_retrieve", "node_evaluate_relevance")

    graph.add_conditional_edges(
        "node_evaluate_relevance",
        route_evaluation_decision,
        {
            "node_final_context": "node_final_context",
            "node_correct_retrieval": "node_correct_retrieval",
        },
    )

    # After correction, re-evaluate candidates
    graph.add_edge("node_correct_retrieval", "node_evaluate_relevance")

    graph.add_edge("node_final_context", END)

    return graph.compile()


# Cached compiled instance
advanced_rag_app = build_advanced_rag_graph()
