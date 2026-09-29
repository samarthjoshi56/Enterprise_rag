"""End-to-end integration tests for Phase 4 Advanced RAG pipeline."""
import pytest
from app.rag.service import run_advanced_rag
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.db.qdrant import init_qdrant_collection
from app.ingestion.service import process_and_ingest_document


@pytest.fixture(scope="module")
async def setup_advanced_rag_data():
    """Ensure DB and Qdrant are initialized with test knowledge."""
    await init_postgres_db()
    await init_qdrant_collection()

    doc_text = (
        "Enterprise Kubernetes Deployment Requirements: "
        "High availability requires deploying worker nodes across at least three availability zones. "
        "Etcd clusters require persistent SSD storage with low IOPS latency. "
        "Production workloads must specify CPU and memory resource requests and limits to prevent OOM kills. "
        "Network policies are mandatory to restrict inter-pod communications and enforce multi-tenancy. "
        "Monitoring is performed via Prometheus and Grafana dashboards with alerting rules."
    ).encode("utf-8")

    async with AsyncSessionLocal() as db:
        await process_and_ingest_document(
            file_bytes=doc_text,
            filename="enterprise_k8s_guide.txt",
            db=db,
        )


@pytest.mark.asyncio
async def test_advanced_rag_e2e_successful_retrieval(setup_advanced_rag_data):
    """
    Verify complete flow:
    Query -> Hybrid Retrieval -> Context Check (YES) -> Rerank -> Final Context
    """
    query = "What are the high availability requirements for enterprise Kubernetes?"

    async with AsyncSessionLocal() as db:
        result = await run_advanced_rag(
            query=query,
            db=db,
            top_k=5,
            top_n=3,
        )

    assert result["should_retrieve"] is True
    assert result["query"] == query
    assert len(result["final_chunks"]) > 0
    assert result["final_chunks"][0].rerank_score is not None
    assert len(result["audit_trail"]) >= 4
    assert result["latency_ms"] > 0

    top_chunk = result["final_chunks"][0]
    assert "Kubernetes" in top_chunk.text or "availability" in top_chunk.text.lower()


@pytest.mark.asyncio
async def test_advanced_rag_e2e_conversational_bypass(setup_advanced_rag_data):
    """Conversational input should trigger Self-RAG [No Retrieve] without errors."""
    query = "Hi there, good morning!"

    async with AsyncSessionLocal() as db:
        result = await run_advanced_rag(
            query=query,
            db=db,
        )

    assert result["should_retrieve"] is False
    assert result["retrieve_token"] == "[No Retrieve]"
    assert result["final_chunks"] == []


@pytest.mark.asyncio
async def test_advanced_rag_e2e_corrective_fallback_branch(setup_advanced_rag_data):
    """
    When given a query that yields poor or ambiguous results,
    the pipeline should trigger CRAG correction and audit trail should record the correction step.
    """
    # Strict threshold forces the correction node to execute
    query = "Kubernetes etcd IOPS latency"

    async with AsyncSessionLocal() as db:
        result = await run_advanced_rag(
            query=query,
            db=db,
            relevance_threshold=15.0,  # Unattainably high threshold triggers CRAG correction
        )

    assert result["correction_count"] >= 1
    assert any("Correction Node" in step for step in result["audit_trail"])
