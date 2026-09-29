"""
SQL Validation and Safety Enforcement for Text2SQL.

Security rules:
1. ONLY single-statement `SELECT` queries are permitted.
2. DDL, DML mutations, administrative commands, and stored procedure executions are strictly rejected.
3. Multiple statements separated by semicolons are strictly rejected.
4. Queries can ONLY reference explicitly allowed tables ('documents', 'document_chunks', 'incidents').
5. System catalogs ('pg_*', 'information_schema') are blocked.
6. Default LIMIT 50 is injected if no LIMIT clause is present.
"""
import re
from typing import Tuple, List, Set, Optional
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlparse.tokens import Keyword, DML

from app.text2sql.schema import ALLOWED_TABLES
from app.core.logging import logger

DANGEROUS_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "EXEC", "EXECUTE", "GRANT", "REVOKE", "COPY", "VACUUM",
    "CALL", "DO", "PREPARE", "DEALLOCATE", "INTO", "MERGE", "REINDEX",
    "EXPLAIN", "LOCK", "REFRESH", "DISCARD"
}


def _extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract referenced table names using regex pattern matching on FROM and JOIN."""
    # Matches: FROM table_name or JOIN table_name (optional schema / alias)
    pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_\"]+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    tables = set()
    for m in matches:
        cleaned = m.strip().strip('"').lower()
        # Strip any schema prefix if present, e.g. public.incidents -> incidents
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        tables.add(cleaned)
    return tables


def validate_sql(sql: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate SQL query for safety.

    Returns:
        (is_valid, error_message_if_invalid, sanitized_sql)
    """
    clean_sql = sql.strip().rstrip(";").strip()
    if not clean_sql:
        return False, "Query cannot be empty", ""

    # 1. Parse statement with sqlparse
    parsed = sqlparse.parse(clean_sql)
    if not parsed:
        return False, "Unable to parse SQL statement", ""

    if len(parsed) > 1:
        return False, "Multi-statement queries are strictly prohibited (SQL injection protection)", ""

    stmt = parsed[0]
    stmt_type = stmt.get_type()

    # 2. Must be SELECT statement
    if stmt_type != "SELECT":
        return False, f"Only SELECT queries are allowed. Detected: {stmt_type or 'UNKNOWN'}", ""

    # 3. Check for dangerous keywords anywhere in the SQL
    tokens = re.findall(r"\b[A-Za-z_]+\b", clean_sql.upper())
    for token in tokens:
        if token in DANGEROUS_KEYWORDS:
            # Special exemption: "SELECT INTO" is dangerous, but "IN" or other benign words are fine
            return False, f"Dangerous SQL keyword detected: '{token}'. Query rejected.", ""

    # 4. Check for system tables or forbidden schemas
    forbidden_prefixes = ["pg_", "information_schema"]
    for prefix in forbidden_prefixes:
        if prefix in clean_sql.lower():
            return False, f"Access to system catalog '{prefix}' is strictly prohibited.", ""

    # 5. Verify all referenced tables are in ALLOWED_TABLES
    referenced_tables = _extract_tables_from_sql(clean_sql)
    if not referenced_tables:
        return False, "Query must reference at least one valid table", ""

    for table in referenced_tables:
        if table not in ALLOWED_TABLES:
            return False, (
                f"Access to table '{table}' is not permitted. "
                f"Allowed tables: {sorted(list(ALLOWED_TABLES))}"
            ), ""

    # 6. Enforce LIMIT clause
    if not re.search(r"\bLIMIT\s+\d+\b", clean_sql, re.IGNORECASE):
        sanitized_sql = f"{clean_sql} LIMIT 50"
    else:
        sanitized_sql = clean_sql

    logger.debug(f"SQL validation passed for: '{sanitized_sql}'")
    return True, None, sanitized_sql
