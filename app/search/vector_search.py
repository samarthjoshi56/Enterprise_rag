"""Vector search: query embedding → Qdrant ANN retrieval."""
from typing import List
from app.db.qdrant import get_async_qdrant_client
from app.ingestion.embeddings import generate_embedding
from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult

settings = get_settings()


async def vector_search(
    query: str,
    top_k: int = None,
) -> List[ChunkResult]:
    """
    Embed the query and perform approximate nearest-neighbour search in Qdrant.

    Returns ChunkResult objects ordered by cosine similarity (highest first),
    with vector_score normalised to [0, 1].
    """
    if top_k is None:
        top_k = settings.VECTOR_SEARCH_TOP_K

    query_vector = generate_embedding(query)
    client = get_async_qdrant_client()

    try:
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
    finally:
        await client.close()

    results: List[ChunkResult] = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            ChunkResult(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index", 0),
                text=payload.get("text", ""),
                vector_score=float(hit.score),
            )
        )

    logger.debug(f"Vector search returned {len(results)} hits for query: '{query[:60]}...'")
    return results
