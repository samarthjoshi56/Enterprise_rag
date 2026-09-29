"""Integration tests for Human Approval flow and SQL execution on PostgreSQL."""
import pytest
from app.db.postgres import init_postgres_db, AsyncSessionLocal
from app.text2sql.service import generate_and_stage_sql, process_approval_and_execute
from app.text2sql.approval import STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED


@pytest.mark.asyncio
async def test_text2sql_approval_flow_rejection():
    """If user rejects the query, execution must NOT occur."""
    await init_postgres_db()

    question = "List all documents"
    staged = await generate_and_stage_sql(question)

    assert staged["status"] == STATUS_PENDING
    query_id = staged["query_id"]
    assert query_id is not None

    async with AsyncSessionLocal() as db:
        result = await process_approval_and_execute(
            query_id=query_id,
            approved=False,
            db=db,
        )

    assert result["status"] == STATUS_REJECTED
    assert result["row_count"] == 0
    assert result["rows"] == []


@pytest.mark.asyncio
async def test_text2sql_approval_flow_execution_top_kubernetes_incidents():
    """
    End-to-end completion condition:
    'What are the top 5 Kubernetes incidents?'
            ↓
      Generated SQL
            ↓
      User approves
            ↓
      PostgreSQL
            ↓
       Results
    """
    await init_postgres_db()

    # Step 1: Generate SQL and stage for human approval
    question = "What are the top 5 Kubernetes incidents?"
    staged = await generate_and_stage_sql(question)

    assert staged["is_safe"] is True
    assert staged["status"] == STATUS_PENDING
    assert "incidents" in staged["sanitized_sql"].lower()
    query_id = staged["query_id"]

    # Step 2: Human inspects generated SQL and explicitly approves
    async with AsyncSessionLocal() as db:
        exec_result = await process_approval_and_execute(
            query_id=query_id,
            approved=True,
            db=db,
        )

    assert exec_result["status"] == STATUS_APPROVED
    assert exec_result["row_count"] > 0
    assert exec_result["columns"] is not None
    assert "title" in exec_result["columns"]
    assert "service" in exec_result["columns"]

    # Verify returned rows match Kubernetes
    first_incident = exec_result["rows"][0]
    assert "kubernetes" in first_incident["service"].lower() or "kubernetes" in first_incident["title"].lower()
