"""Unit tests for Natural Language to SQL generation."""
import pytest
from app.text2sql.generator import generate_sql_query
from app.text2sql.validator import validate_sql


@pytest.mark.asyncio
async def test_generate_sql_top_kubernetes_incidents():
    """Verify text2sql converts question into a valid Kubernetes incident SELECT query."""
    question = "What are the top 5 Kubernetes incidents?"
    sql, explanation = await generate_sql_query(question)

    assert "SELECT" in sql.upper()
    assert "incidents" in sql.lower()
    assert "kubernetes" in sql.lower()

    # Must pass validation
    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is True
    assert err is None


@pytest.mark.asyncio
async def test_generate_sql_aggregation_severity():
    """Verify group by aggregation query is generated correctly."""
    question = "Show me the count of incidents by severity"
    sql, explanation = await generate_sql_query(question)

    assert "SELECT" in sql.upper()
    assert "COUNT" in sql.upper()
    assert "severity" in sql.lower()

    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is True


@pytest.mark.asyncio
async def test_generate_sql_documents_count():
    """Verify document metadata queries are generated correctly."""
    question = "How many documents are ingested?"
    sql, explanation = await generate_sql_query(question)

    assert "SELECT" in sql.upper()
    assert "documents" in sql.lower()

    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is True
