"""
SQL Query Executor for Text2SQL.

Executes validated and human-approved SQL queries on PostgreSQL
using read-only transactions and structured result mapping.
"""
import time
from datetime import datetime, date
from typing import Dict, Any, List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger


def _serialize_value(val: Any) -> Any:
    """Format non-JSON serializable values like datetimes into ISO strings."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return val


async def execute_approved_sql(
    sql: str,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    Execute an approved SELECT query against the PostgreSQL database.

    Returns:
        {
            "columns": List[str],
            "rows": List[Dict[str, Any]],
            "row_count": int,
            "execution_time_ms": float,
        }
    """
    start_time = time.perf_counter()

    try:
        # Enforce read-only execution at session level
        await db.execute(text("SET TRANSACTION READ ONLY;"))
        result = await db.execute(text(sql))

        # Extract column names and row mappings
        columns = list(result.keys()) if result.returns_rows else []
        raw_rows = result.mappings().all() if result.returns_rows else []

        # Convert rows to serializable dicts
        rows: List[Dict[str, Any]] = [
            {k: _serialize_value(v) for k, v in row.items()}
            for row in raw_rows
        ]

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"Text2SQL execution succeeded: {len(rows)} rows returned in {execution_time_ms}ms"
        )

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": execution_time_ms,
        }
    except Exception as e:
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"Text2SQL execution failed after {execution_time_ms}ms: {e}")
        raise RuntimeError(f"Database query execution failed: {str(e)}") from e
