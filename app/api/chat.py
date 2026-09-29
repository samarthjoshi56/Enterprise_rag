import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.db.postgres import get_async_db
from app.search.vector_search import vector_search
from app.search.keyword_search import keyword_search
from app.search.hybrid import fuse_results
from app.search.reranker import rerank
from app.rag.llm import generate_rag_answer
from app.rag.graph import advanced_rag_app
from app.cache.service import cache_service, normalize_cache_key

router = APIRouter(prefix="/chat", tags=["Chat"])
settings = get_settings()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question / prompt")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    use_advanced_rag: bool = Field(default=False, description="Use HyDE + CRAG + Self-RAG graph pipeline")


class SourceItem(BaseModel):
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    latency_ms: float
    cached: bool
    mode: str
    audit_trail: Optional[List[str]] = None


@router.post("", response_model=ChatResponse, summary="Answer user questions grounded in document context")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_async_db)):
    start_time = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")

    cache_key = normalize_cache_key("chat", query, adv=request.use_advanced_rag, top_n=request.top_n)
    cached_data = await cache_service.get(cache_key)
    if cached_data:
        cached_data["cached"] = True
        return cached_data


    audit_trail = []
    sources: List[SourceItem] = []
    context_chunks = []

    try:
        if request.use_advanced_rag:
            # Run LangGraph Advanced RAG pipeline
            initial_state = {
                "query": query,
                "should_retrieve": True,
                "retrieve_token": "[Retrieve]",
                "hypothetical_doc": None,
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "retrieval_grade": "CORRECT",
                "correction_needed": False,
                "correction_count": 0,
                "rewritten_query": None,
                "final_chunks": [],
                "audit_trail": [],
            }
            state = await advanced_rag_app.ainvoke(
                initial_state,
                config={"configurable": {"db": db, "top_k": settings.VECTOR_SEARCH_TOP_K}},
            )
            final_chunks = state.get("final_chunks", [])[:request.top_n]
            audit_trail = state.get("audit_trail", [])
            for c in final_chunks:
                context_chunks.append({
                    "text": c.text,
                    "filename": c.filename or "unknown",
                    "page_number": c.page_number,
                })
                s_val = c.rerank_score if c.rerank_score is not None else c.hybrid_score
                sources.append(
                    SourceItem(
                        filename=c.filename or "unknown",
                        page_number=c.page_number,
                        chunk_index=c.chunk_index,
                        score=round(float(s_val), 4),
                        text=c.text,
                    )
                )
            mode = "advanced_rag (HyDE + CRAG + Self-RAG)"
        else:
            # Standard Hybrid Search (Vector + FTS) + Cross-Encoder Rerank
            vec_res = await vector_search(query, top_k=settings.VECTOR_SEARCH_TOP_K)
            kw_res = await keyword_search(query, db=db, top_k=settings.KEYWORD_SEARCH_TOP_K)
            fused = fuse_results(vec_res, kw_res, alpha=settings.HYBRID_ALPHA)
            reranked = rerank(query, fused, top_n=request.top_n)

            for c in reranked:
                context_chunks.append({
                    "text": c.text,
                    "filename": c.filename or "unknown",
                    "page_number": c.page_number,
                })
                s_val = c.rerank_score if c.rerank_score is not None else c.hybrid_score
                sources.append(
                    SourceItem(
                        filename=c.filename or "unknown",
                        page_number=c.page_number,
                        chunk_index=c.chunk_index,
                        score=round(float(s_val), 4),
                        text=c.text,
                    )
                )
            mode = "hybrid_search + cross_encoder_rerank"


        # Generate answer from LLM grounded in context
        answer = await generate_rag_answer(query, context_chunks)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response_dict = {
            "query": query,
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "latency_ms": latency_ms,
            "cached": False,
            "mode": mode,
            "audit_trail": audit_trail if request.use_advanced_rag else None,
        }

        # Cache response in Redis
        await cache_service.set(cache_key, response_dict)
        return response_dict


    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation error: {str(e)}",
        )
