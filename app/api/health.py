import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.db.postgres import check_postgres_connection
from app.db.qdrant import check_qdrant_connection
from app.db.redis import check_redis_connection

router = APIRouter(tags=["Health"])
settings = get_settings()


def verify_framework_imports() -> Dict[str, Any]:
    """Verify LangChain and LangGraph imports and return their versions."""
    results = {}
    
    try:
        import langchain
        results["langchain"] = {
            "status": "available",
            "version": getattr(langchain, "__version__", "unknown")
        }
    except Exception as e:
        results["langchain"] = {
            "status": "error",
            "error": str(e)
        }
        
    try:
        import langgraph
        results["langgraph"] = {
            "status": "available",
            "version": getattr(langgraph, "__version__", "unknown")
        }
    except Exception as e:
        results["langgraph"] = {
            "status": "error",
            "error": str(e)
        }

    try:
        import langchain_core
        results["langchain_core"] = {
            "status": "available",
            "version": getattr(langchain_core, "__version__", "unknown")
        }
    except Exception as e:
        results["langchain_core"] = {
            "status": "error",
            "error": str(e)
        }

    return results


@router.get("/health", summary="Basic and detailed system health check")
@router.get("/health/liveness", summary="Liveness probe")
async def health_check():
    """
    Health check endpoint verifying:
    - FastAPI application status
    - PostgreSQL connectivity
    - Qdrant vector database connectivity
    - Redis cache connectivity
    - LangChain & LangGraph framework imports
    """
    # Execute database checks concurrently
    pg_task = check_postgres_connection()
    qdrant_task = check_qdrant_connection()
    redis_task = check_redis_connection()

    pg_result, qdrant_result, redis_result = await asyncio.gather(
        pg_task, qdrant_task, redis_task
    )

    frameworks = verify_framework_imports()

    # Determine overall status
    services = [pg_result, qdrant_result, redis_result]
    all_connected = all(s.get("status") == "connected" for s in services)
    any_error = any(s.get("status") == "error" for s in services)

    if all_connected:
        overall_status = "healthy"
        http_status_code = status.HTTP_200_OK
    elif any_error:
        overall_status = "degraded"
        http_status_code = status.HTTP_200_OK  # Still 200 to allow inspection, or HTTP_503 if strict
    else:
        overall_status = "unhealthy"
        http_status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    payload = {
        "status": overall_status,
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "postgres": pg_result,
            "qdrant": qdrant_result,
            "redis": redis_result,
        },
        "frameworks": frameworks,
    }

    return JSONResponse(status_code=http_status_code, content=payload)
