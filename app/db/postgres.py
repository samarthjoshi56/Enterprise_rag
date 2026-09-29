import time
from typing import AsyncGenerator, Dict, Any
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

engine: AsyncEngine = create_async_engine(
    settings.async_postgres_url,
    echo=False,
    future=True,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_postgres_db() -> None:
    """Initialize database tables on application startup."""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Ensure chunk_text column exists for existing tables created before Phase 3
        await conn.execute(
            text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_text TEXT;")
        )
        # Create full-text search GIN index if it doesn't already exist
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_fts "
                "ON document_chunks USING gin(to_tsvector('english', COALESCE(chunk_text, '')));"
            )
        )
        # Check if incidents table has records; if not, seed realistic enterprise incidents
        res = await conn.execute(text("SELECT COUNT(*) FROM incidents;"))
        count = res.scalar()
        if count == 0:
            await conn.execute(
                text(
                    """
                    INSERT INTO incidents (id, title, service, severity, status, impact_summary, created_at)
                    VALUES
                    ('inc-001', 'Kubernetes Node NotReady - Disk Pressure in us-east-1', 'kubernetes', 'CRITICAL', 'RESOLVED', 'Worker node failed due to full root volume.', NOW() - INTERVAL '5 days'),
                    ('inc-002', 'Kubernetes Pod CrashLoopBackOff on Ingress Controller', 'kubernetes', 'HIGH', 'RESOLVED', 'ConfigMap syntax error caused nginx ingress restart loops.', NOW() - INTERVAL '4 days'),
                    ('inc-003', 'Kubernetes CoreDNS Latency Spike', 'kubernetes', 'HIGH', 'RESOLVED', 'DNS resolution throttled due to conntrack table exhaustion.', NOW() - INTERVAL '3 days'),
                    ('inc-004', 'Kubernetes HPA Thrashing on Payment Service', 'kubernetes', 'MEDIUM', 'RESOLVED', 'CPU scaling thresholds were set too aggressively.', NOW() - INTERVAL '2 days'),
                    ('inc-005', 'Kubernetes OOMKilled on Metrics Server Pod', 'kubernetes', 'LOW', 'RESOLVED', 'Memory limit increased from 200Mi to 500Mi.', NOW() - INTERVAL '1 day'),
                    ('inc-006', 'PostgreSQL Connection Limit Exceeded', 'postgresql', 'CRITICAL', 'RESOLVED', 'Client application connection leak exhausted pool.', NOW() - INTERVAL '6 days');
                    """
                )
            )


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session in FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_postgres_connection() -> Dict[str, Any]:
    """Verify PostgreSQL connectivity and return operational health stats."""
    start_time = time.perf_counter()
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT version();"))
            pg_version = result.scalar()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            return {
                "status": "connected",
                "latency_ms": latency_ms,
                "version": pg_version.split()[0] if pg_version else "PostgreSQL",
                "details": "Connection successful",
            }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"PostgreSQL connection check failed: {e}")
        return {
            "status": "error",
            "latency_ms": latency_ms,
            "error": str(e),
        }
