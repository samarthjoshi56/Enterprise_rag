#!/usr/bin/env python3
"""CLI verification script for Phase 4 — Advanced RAG (HyDE, CRAG, Self-RAG, LangGraph)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.db.qdrant import init_qdrant_collection
from app.rag.hyde import hyde_search, compare_retrieval
from app.rag.crag import evaluate_retrieval_quality, execute_corrective_retrieval
from app.rag.self_rag import (
    should_retrieve,
    evaluate_chunk_relevance,
    validate_answer_support,
)
from app.rag.service import run_advanced_rag
from app.search.vector_search import vector_search

QUERY = "Kubernetes deployment requirements"


async def main():
    print("=" * 70)
    print(" Enterprise RAG — Phase 4 Advanced RAG Verification")
    print("=" * 70)

    await init_postgres_db()
    await init_qdrant_collection()

    # -------------------------------------------------------------
    # 1. HyDE (Hypothetical Document Embeddings)
    # -------------------------------------------------------------
    print("\n[1] Testing HyDE (Hypothetical Document Embeddings):")
    normal_results = await vector_search(QUERY, top_k=5)
    hyde_results, hypo_doc = await hyde_search(QUERY, top_k=5)

    print(f"    Generated Hypothetical Document:\n    \"{hypo_doc[:120]}...\"")
    print(f"    HyDE Vector Hits: {len(hyde_results)}")
    comparison = compare_retrieval(normal_results, hyde_results)
    print(f"    Comparison with standard retrieval:")
    print(f"      - Common chunk count  : {comparison['overlap_count']}")
    print(f"      - Jaccard similarity  : {comparison['jaccard_similarity']}")
    print(f"      - HyDE unique chunks  : {len(comparison['hyde_only_ids'])}")

    # -------------------------------------------------------------
    # 2. CRAG (Corrective RAG)
    # -------------------------------------------------------------
    print("\n[2] Testing CRAG (Corrective RAG):")
    grade, refined = evaluate_retrieval_quality(QUERY, hyde_results)
    print(f"    Retrieval Quality Grade : {grade}")
    print(f"    Refined chunks count    : {len(refined)}/{len(hyde_results)}")

    async with AsyncSessionLocal() as db:
        improved, rewritten = await execute_corrective_retrieval(
            query="Can you please explain the k8s deployment requirements in detail?",
            db=db,
            reason="testing query rewrite",
            top_k=3,
        )
    print(f"    Query Rewriter test:")
    print(f"      Original  : 'Can you please explain the k8s deployment requirements in detail?'")
    print(f"      Rewritten : '{rewritten}'")
    print(f"      Corrective candidates retrieved: {len(improved)}")

    # -------------------------------------------------------------
    # 3. Self-RAG (Self-Reflective RAG)
    # -------------------------------------------------------------
    print("\n[3] Testing Self-RAG Reflection Tokens:")
    # [Retrieve] token
    _, tok1 = should_retrieve("What are the Kubernetes deployment requirements?")
    _, tok2 = should_retrieve("Good morning! How are you doing?")
    print(f"    Token [Retrieve] for factual query       : {tok1}")
    print(f"    Token [Retrieve] for conversational text : {tok2}")

    # [IsREL] token
    if refined:
        is_rel, rel_tok, score = evaluate_chunk_relevance(QUERY, refined[0])
        print(f"    Token [IsREL] for top chunk (score={score:.4f}): {rel_tok}")

    # [IsSUP] token
    grounded_stmt = "Kubernetes deployments require a cluster and kubectl manifests."
    hallucinated_stmt = "Kubernetes runs directly on quantum computers using quantum entanglement."
    is_sup1, sup_tok1 = await validate_answer_support(refined, grounded_stmt)
    is_sup2, sup_tok2 = await validate_answer_support(refined, hallucinated_stmt)
    print(f"    Token [IsSUP] for grounded statement     : {sup_tok1}")
    print(f"    Token [IsSUP] for hallucinated statement : {sup_tok2}")

    # -------------------------------------------------------------
    # 4. LangGraph End-to-End Workflow
    # -------------------------------------------------------------
    print("\n[4] Testing Compiled LangGraph StateGraph Workflow:")
    async with AsyncSessionLocal() as db:
        rag_output = await run_advanced_rag(
            query=QUERY,
            db=db,
            top_k=5,
            top_n=3,
        )

    print(f"    Query             : \"{rag_output['query']}\"")
    print(f"    Retrieval Decision: {rag_output['retrieve_token']}")
    print(f"    CRAG Grade        : {rag_output['retrieval_grade']}")
    print(f"    Corrections Made  : {rag_output['correction_count']}")
    print(f"    Final Context Chunks: {len(rag_output['final_chunks'])}")
    print(f"    Total Latency     : {rag_output['latency_ms']} ms")

    print("\n    Graph Execution Audit Trail:")
    for step in rag_output["audit_trail"]:
        print(f"      -> {step}")

    if rag_output["final_chunks"]:
        print("\n    Top Final Context Chunk:")
        top = rag_output["final_chunks"][0]
        print(f"      Score : rerank={top.rerank_score:.4f} | source={top.filename} p{top.page_number}")
        print(f"      Text  : {top.text[:120]}...")

    print("\n" + "=" * 70)
    print(" PHASE 4 ADVANCED RAG VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
