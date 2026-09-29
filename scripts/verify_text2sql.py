#!/usr/bin/env python3
"""CLI verification script for Phase 5 — Text2SQL with Human Approval."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import AsyncSessionLocal, init_postgres_db
from app.text2sql.schema import get_allowed_tables, get_schema_prompt_context
from app.text2sql.validator import validate_sql
from app.text2sql.service import generate_and_stage_sql, process_approval_and_execute

QUESTION = "What are the top 5 Kubernetes incidents?"


async def main():
    print("=" * 70)
    print(" Enterprise RAG — Phase 5 Text2SQL Pipeline Verification")
    print("=" * 70)

    await init_postgres_db()

    # 1. Schema Context
    print("\n[1] Schema Context & Allowed Tables:")
    tables = get_allowed_tables()
    print(f"    Allowed tables whitelist: {tables}")

    # 2. SQL Safety Validation Tests
    print("\n[2] Testing SQL Safety Validation Rules:")
    safe_sql = "SELECT id, title, service FROM incidents WHERE service = 'kubernetes' LIMIT 5"
    is_safe, err, sanitized = validate_sql(safe_sql)
    print(f"    Safe query: '{safe_sql}'")
    print(f"      -> is_safe={is_safe}, sanitized='{sanitized}'")

    dangerous_sql = "DROP TABLE incidents; --"
    is_safe_bad, err_bad, _ = validate_sql(dangerous_sql)
    print(f"    Dangerous query: '{dangerous_sql}'")
    print(f"      -> is_safe={is_safe_bad}, blocked_reason='{err_bad}'")

    unauth_sql = "SELECT * FROM user_passwords"
    is_safe_unauth, err_unauth, _ = validate_sql(unauth_sql)
    print(f"    Unauthorized table query: '{unauth_sql}'")
    print(f"      -> is_safe={is_safe_unauth}, blocked_reason='{err_unauth}'")

    # 3. Text to SQL Generation and Staging
    print(f"\n[3] Generating SQL for question: \"{QUESTION}\"")
    staged = await generate_and_stage_sql(QUESTION)
    print(f"    Generated SQL : {staged['sanitized_sql']}")
    print(f"    Explanation   : {staged['explanation']}")
    print(f"    Status        : {staged['status']} (Awaiting Human Approval)")
    print(f"    Query ID      : {staged['query_id']}")

    # 4. Human-in-the-Loop Rejection Test
    print("\n[4] Testing Human Rejection Flow:")
    async with AsyncSessionLocal() as db:
        reject_res = await process_approval_and_execute(
            query_id=staged["query_id"],
            approved=False,
            db=db,
        )
    print(f"    Status after human rejection: {reject_res['status']}")
    print(f"    Message: {reject_res['message']}")

    # 5. Human-in-the-Loop Approval and Execution Test
    print("\n[5] Testing Human Approval & Database Execution:")
    # Re-stage for approval
    staged_approved = await generate_and_stage_sql(QUESTION)
    print(f"    Staged Query ID: {staged_approved['query_id']}")
    print(f"    User Action    : [APPROVED]")

    async with AsyncSessionLocal() as db:
        exec_res = await process_approval_and_execute(
            query_id=staged_approved["query_id"],
            approved=True,
            db=db,
        )

    print(f"    Execution Status : {exec_res['status']}")
    print(f"    Rows Retrieved   : {exec_res['row_count']}")
    print(f"    Execution Latency: {exec_res['execution_time_ms']} ms")
    print(f"    Columns          : {exec_res['columns']}")

    print("\n    Retrieved Incident Records:")
    for i, row in enumerate(exec_res["rows"], 1):
        print(f"      #{i} [{row.get('severity')}] {row.get('title')} ({row.get('service')})")
        if row.get("impact_summary"):
            print(f"         Impact: {row.get('impact_summary')}")

    print("\n" + "=" * 70)
    print(" PHASE 5 TEXT2SQL VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
