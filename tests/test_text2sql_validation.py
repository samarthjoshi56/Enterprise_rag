"""Unit tests for SQL validation and safety enforcement."""
import pytest
from app.text2sql.validator import validate_sql


def test_validate_safe_select_query():
    """Simple SELECT queries on allowed tables must pass validation."""
    sql = "SELECT id, title, severity FROM incidents WHERE service = 'kubernetes' LIMIT 5"
    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is True
    assert err is None
    assert "LIMIT 5" in sanitized


def test_validate_auto_inject_limit():
    """Queries without a LIMIT clause must automatically receive LIMIT 50."""
    sql = "SELECT id, filename FROM documents"
    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is True
    assert "LIMIT 50" in sanitized


@pytest.mark.parametrize(
    "dangerous_sql",
    [
        "DROP TABLE incidents",
        "DELETE FROM incidents WHERE id = '1'",
        "UPDATE incidents SET status = 'CLOSED'",
        "INSERT INTO incidents (title) VALUES ('Hacked')",
        "TRUNCATE TABLE documents",
        "ALTER TABLE documents DROP COLUMN filename",
        "CREATE TABLE backdoor (id INT)",
        "GRANT ALL PRIVILEGES ON DATABASE enterprise_rag TO attacker",
    ],
)
def test_validate_reject_mutations_and_ddl(dangerous_sql):
    """Mutations, DDL, and permission grants must be strictly rejected."""
    is_safe, err, sanitized = validate_sql(dangerous_sql)
    assert is_safe is False
    assert err is not None
    assert "rejected" in err.lower() or "only select" in err.lower()


def test_validate_reject_multi_statement_sql_injection():
    """Multi-statement SQL injection attempts must be blocked."""
    sql = "SELECT * FROM incidents; DROP TABLE documents;"
    is_safe, err, sanitized = validate_sql(sql)
    assert is_safe is False
    assert "multi-statement" in err.lower()


def test_validate_reject_unauthorized_tables():
    """Accessing tables outside the allowed whitelist must be rejected."""
    unauthorized = "SELECT * FROM user_credentials WHERE id = 1"
    is_safe, err, sanitized = validate_sql(unauthorized)
    assert is_safe is False
    assert "not permitted" in err.lower()


def test_validate_reject_system_catalogs():
    """Accessing PostgreSQL internal pg_catalog or information_schema must be rejected."""
    sys_query = "SELECT * FROM pg_catalog.pg_tables"
    is_safe, err, sanitized = validate_sql(sys_query)
    assert is_safe is False
    assert "system catalog" in err.lower() or "not permitted" in err.lower()
