"""
Text2SQL Orchestration Service.

Connects:
1. SQL Generation
2. Security Validation
3. Human-in-the-Loop Approval Staging
4. Safe Database Execution
"""
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.text2sql.generator import generate_sql_query
from app.text2sql.validator import validate_sql
from app.text2sql.schema import get_allowed_tables
from app.text2sql.approval import (
    create_approval_request,
    get_approval_request,
    approve_request,
    reject_request,
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_REJECTED,
)
from app.text2sql.executor import execute_approved_sql
from app.core.logging import logger


async def generate_and_stage_sql(question: str) -> Dict[str, Any]:
    """
    Step 1: Convert natural language to SQL, validate security,
    and stage query for human approval.
    """
    raw_sql, explanation = await generate_sql_query(question)

    # Validate SQL
    is_safe, error_msg, sanitized_sql = validate_sql(raw_sql)
    if not is_safe:
        return {
            "query_id": None,
            "question": question,
            "generated_sql": raw_sql,
            "sanitized_sql": None,
            "explanation": explanation,
            "is_safe": False,
            "error": error_msg,
            "status": "VALIDATION_FAILED",
            "allowed_tables": get_allowed_tables(),
        }

    # Stage approval request
    req = create_approval_request(
        question=question,
        sql=sanitized_sql,
        explanation=explanation,
    )

    logger.info(f"Staged Text2SQL query_id={req.query_id} for approval. SQL: '{sanitized_sql}'")

    return {
        "query_id": req.query_id,
        "question": question,
        "generated_sql": raw_sql,
        "sanitized_sql": sanitized_sql,
        "explanation": explanation,
        "is_safe": True,
        "error": None,
        "status": req.status,
        "allowed_tables": get_allowed_tables(),
    }


async def process_approval_and_execute(
    query_id: str,
    approved: bool,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    Step 2: Process human approval decision. If approved, execute SQL.
    """
    req = get_approval_request(query_id)
    if not req:
        raise ValueError(f"Query request with ID '{query_id}' not found.")

    if not approved:
        reject_request(query_id)
        logger.info(f"Text2SQL query_id={query_id} was REJECTED by human approver.")
        return {
            "query_id": query_id,
            "status": STATUS_REJECTED,
            "message": "Query was rejected and will not be executed.",
            "sql": req.generated_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0.0,
        }

    # Mark as approved
    ok, req, msg = approve_request(query_id)
    if not ok:
        raise ValueError(msg)

    # Re-validate immediately before execution as a defense-in-depth guarantee
    is_safe, err, executable_sql = validate_sql(req.generated_sql)
    if not is_safe:
        raise ValueError(f"Security validation failed prior to execution: {err}")

    # Execute against database
    exec_result = await execute_approved_sql(executable_sql, db)

    return {
        "query_id": query_id,
        "status": STATUS_APPROVED,
        "message": "Query approved and executed successfully.",
        "sql": executable_sql,
        "columns": exec_result["columns"],
        "rows": exec_result["rows"],
        "row_count": exec_result["row_count"],
        "execution_time_ms": exec_result["execution_time_ms"],
    }
