"""
FastAPI endpoints for Text2SQL with Human-in-the-Loop (HITL) approval.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_async_db
from app.text2sql.service import generate_and_stage_sql, process_approval_and_execute
from app.text2sql.schema import SCHEMA_METADATA, get_allowed_tables

router = APIRouter(prefix="/text2sql", tags=["Text2SQL"])


# --- Request and Response Schemas ---

class Text2SQLGenerateRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What are the top 5 Kubernetes incidents?"],
        description="Natural language question to convert into SQL",
    )


class Text2SQLGenerateResponse(BaseModel):
    query_id: Optional[str]
    question: str
    generated_sql: str
    sanitized_sql: Optional[str]
    explanation: str
    is_safe: bool
    error: Optional[str]
    status: str
    allowed_tables: List[str]


class Text2SQLExecuteRequest(BaseModel):
    query_id: str = Field(..., description="Unique query ID from the generation step")
    approved: bool = Field(..., description="Explicit human approval: True to execute, False to reject")


class Text2SQLExecuteResponse(BaseModel):
    query_id: str
    status: str
    message: str
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float


class Text2SQLDirectRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    approved: bool = Field(
        default=False,
        description="If True, proceeds to execute immediately if safe. If False, stages for review.",
    )


# --- Endpoints ---

@router.get(
    "/schema",
    summary="Get Database Schema and Allowed Tables",
    description="Returns the tables, columns, and descriptions available for Text2SQL queries.",
)
async def get_schema():
    """Retrieve allowed tables and schema metadata."""
    return {
        "allowed_tables": get_allowed_tables(),
        "schema": SCHEMA_METADATA,
    }


@router.post(
    "/generate",
    response_model=Text2SQLGenerateResponse,
    summary="Generate SQL from Natural Language (Step 1)",
    description="Converts natural language into SQL, validates for security, and stages for human approval.",
)
async def generate_sql_endpoint(request: Text2SQLGenerateRequest):
    """Generate SQL and stage for human approval."""
    result = await generate_and_stage_sql(request.question)
    return Text2SQLGenerateResponse(**result)


@router.post(
    "/execute",
    response_model=Text2SQLExecuteResponse,
    summary="Approve and Execute Staged SQL (Step 2)",
    description="Requires explicit human approval (approved=True) to execute the staged query on PostgreSQL.",
)
async def execute_sql_endpoint(
    request: Text2SQLExecuteRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Execute a staged SQL query upon human approval."""
    try:
        result = await process_approval_and_execute(
            query_id=request.query_id,
            approved=request.approved,
            db=db,
        )
        return Text2SQLExecuteResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error: {str(e)}",
        )


@router.post(
    "",
    summary="Combined Text2SQL Query Endpoint",
    description="Convenience endpoint: generates SQL and optionally executes if approved=True.",
)
async def text2sql_direct_endpoint(
    request: Text2SQLDirectRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """All-in-one Text2SQL generation and conditional execution."""
    stage_result = await generate_and_stage_sql(request.question)

    if not stage_result["is_safe"]:
        return stage_result

    if not request.approved:
        return stage_result

    # If approved=True, execute the query
    exec_result = await process_approval_and_execute(
        query_id=stage_result["query_id"],
        approved=True,
        db=db,
    )
    return {
        **stage_result,
        "execution": exec_result,
    }
