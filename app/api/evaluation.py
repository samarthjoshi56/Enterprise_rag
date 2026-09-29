"""
FastAPI endpoints for RAG Evaluation and Retrieval Comparison.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_async_db
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.runner import run_retrieval_evaluation
from app.evaluation.comparison import compare_retrieval_configurations
from app.search.service import hybrid_search

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationRunRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20, description="Top-K cutoff for evaluation metrics")


class EvaluationCompareRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20, description="Top-K cutoff for comparative benchmark")


@router.get(
    "/dataset",
    summary="Get Evaluation Benchmark Dataset",
    description="Returns the standard evaluation test cases with ground-truth keywords and target services.",
)
async def get_dataset():
    """Retrieve the evaluation benchmark dataset."""
    samples = load_evaluation_dataset()
    return {
        "total_samples": len(samples),
        "samples": [
            {
                "id": s.id,
                "question": s.question,
                "ground_truth_keywords": s.ground_truth_keywords,
                "expected_answer": s.expected_answer,
                "target_service": s.target_service,
            }
            for s in samples
        ],
    }


@router.post(
    "/run",
    summary="Run Evaluation on Hybrid Search Pipeline",
    description="Runs the standard evaluation dataset against the hybrid retrieval pipeline and calculates Precision@K, Recall@K, MRR, and Hit Rate.",
)
async def run_evaluation_endpoint(
    request: EvaluationRunRequest = EvaluationRunRequest(),
    db: AsyncSession = Depends(get_async_db),
):
    """Execute evaluation benchmark against the active search pipeline."""
    async def retriever(q: str):
        res = await hybrid_search(q, db=db, top_k=request.k, bypass_cache=True)
        return [c.text for c in res.get("results", [])]

    report = await run_retrieval_evaluation(retriever_func=retriever, k=request.k)
    return report


@router.post(
    "/compare",
    summary="Compare Retrieval Configurations",
    description="Benchmarks Vector-Only, Keyword-Only, Hybrid, Hybrid + Reranking, and Advanced RAG side-by-side.",
)
async def compare_retrieval_endpoint(
    request: EvaluationCompareRequest = EvaluationCompareRequest(),
    db: AsyncSession = Depends(get_async_db),
):
    """Run comparative evaluation across different retrieval architectures."""
    comparison = await compare_retrieval_configurations(db=db, k=request.k)
    return comparison
