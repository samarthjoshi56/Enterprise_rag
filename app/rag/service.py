"""
Advanced RAG orchestration service.
Executes the compiled LangGraph workflow containing HyDE, CRAG, and Self-RAG.
"""
import time
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.rag.graph import advanced_rag_app, AdvancedRAGState

settings = get_settings()


async def run_advanced_rag(
    query: str,
    db: Optional[AsyncSession] = None,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute the end-to-end Advanced RAG pipeline.

    Flow:
    1. Self-RAG [Retrieve] query evaluation
    2. Hybrid Search + HyDE retrieval
    3. CRAG quality grading + Self-RAG [IsREL] filtering
    4. Corrective retrieval with query reformulation (if context is poor)
    5. Cross-Encoder reranking -> Final best context

    Returns:
        Structured result dict with final_chunks, grades, audit_trail, and latency.
    """
    start = time.perf_counter()

    initial_state: AdvancedRAGState = {
        "query": query,
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

    config = {
        "configurable": {
            "db": db,
            "top_k": top_k or settings.VECTOR_SEARCH_TOP_K,
            "top_n": top_n or settings.RERANKER_TOP_N,
            "relevance_threshold": (
                relevance_threshold
                if relevance_threshold is not None
                else settings.CRAG_RELEVANCE_THRESHOLD
            ),
        }
    }

    result_state = await advanced_rag_app.ainvoke(initial_state, config=config)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f"Advanced RAG completed for query='{query[:50]}' | "
        f"final_chunks={len(result_state.get('final_chunks', []))} | "
        f"grade={result_state.get('retrieval_grade')} | "
        f"corrections={result_state.get('correction_count')} | "
        f"latency={latency_ms}ms"
    )

    return {
        "query": query,
        "should_retrieve": result_state.get("should_retrieve", True),
        "retrieve_token": result_state.get("retrieve_token", ""),
        "retrieval_grade": result_state.get("retrieval_grade", ""),
        "rewritten_query": result_state.get("rewritten_query"),
        "hypothetical_doc": result_state.get("hypothetical_doc"),
        "correction_count": result_state.get("correction_count", 0),
        "final_chunks": result_state.get("final_chunks", []),
        "audit_trail": result_state.get("audit_trail", []),
        "latency_ms": latency_ms,
    }
