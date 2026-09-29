"""Keyword search: PostgreSQL full-text search using to_tsquery / plainto_tsquery."""
from typing import List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult

settings = get_settings()

# SQL: rank by ts_rank of a GIN tsvector against the query, join document for filename.
# We use plainto_tsquery for a safe, operator-free keyword search.
_FTS_SQL = text(
    """
    SELECT
        dc.id           AS chunk_id,
        dc.document_id,
        d.filename,
        dc.page_number,
        dc.chunk_index,
        dc.chunk_text   AS text,
        ts_rank(
            to_tsvector('english', COALESCE(dc.chunk_text, '')),
            plainto_tsquery('english', :query)
        ) AS fts_rank
    FROM document_chunks dc
    JOIN documents d ON d.id = dc.document_id
    WHERE
        dc.chunk_text IS NOT NULL
        AND to_tsvector('english', COALESCE(dc.chunk_text, ''))
            @@ plainto_tsquery('english', :query)
    ORDER BY fts_rank DESC
    LIMIT :limit
    """
)


async def keyword_search(
    query: str,
    db: AsyncSession,
    top_k: int = None,
) -> List[ChunkResult]:
    """
    Full-text search over document_chunks using PostgreSQL tsvector/tsquery.

    Returns ChunkResult objects ordered by ts_rank (highest first).
    Scores are normalised to [0, 1] relative to the highest rank in the result set.
    """
    if top_k is None:
        top_k = settings.KEYWORD_SEARCH_TOP_K

    rows = (
        await db.execute(_FTS_SQL, {"query": query, "limit": top_k})
    ).mappings().all()

    if not rows:
        logger.debug(f"Keyword search returned 0 hits for query: '{query[:60]}'")
        return []

    # Normalise scores relative to top hit
    max_rank = float(rows[0]["fts_rank"]) if rows else 1.0
    if max_rank == 0.0:
        max_rank = 1.0

    results: List[ChunkResult] = []
    for row in rows:
        results.append(
            ChunkResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                filename=row["filename"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"] or "",
                keyword_score=float(row["fts_rank"]) / max_rank,
            )
        )

    logger.debug(f"Keyword search returned {len(results)} hits for query: '{query[:60]}'")
    return results
