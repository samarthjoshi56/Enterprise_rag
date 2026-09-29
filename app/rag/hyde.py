"""
HyDE: Hypothetical Document Embeddings (Gao et al., 2022).

Flow:
1. Generate hypothetical answer / document from query.
2. Embed hypothetical document using embedding model.
3. Search Qdrant ANN index using the hypothetical vector.
4. Compare and fuse with normal retrieval.
"""
from typing import List, Tuple, Dict, Any
from app.core.config import get_settings
from app.core.logging import logger
from app.db.qdrant import get_async_qdrant_client
from app.ingestion.embeddings import generate_embedding
from app.search.models import ChunkResult
from app.rag.llm import generate_hypothetical_document

settings = get_settings()


async def hyde_search(
    query: str,
    top_k: int = None,
) -> Tuple[List[ChunkResult], str]:
    """
    Execute HyDE retrieval:
    1. Generate hypothetical answering passage.
    2. Embed the passage into the dense embedding space.
    3. Retrieve nearest neighbour points from Qdrant.

    Returns:
        (results, hypothetical_document_text)
    """
    if top_k is None:
        top_k = settings.VECTOR_SEARCH_TOP_K

    # Step 1: Generate hypothetical document
    hypo_doc = await generate_hypothetical_document(query)
    logger.debug(f"HyDE generated hypothetical document ({len(hypo_doc)} chars)")

    # Step 2: Embed hypothetical document
    hypo_vector = generate_embedding(hypo_doc)

    # Step 3: Search Qdrant using the hypothetical vector
    client = get_async_qdrant_client()
    try:
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=hypo_vector,
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

    logger.debug(f"HyDE search returned {len(results)} hits")
    return results, hypo_doc


def compare_retrieval(
    normal_results: List[ChunkResult],
    hyde_results: List[ChunkResult],
) -> Dict[str, Any]:
    """
    Compare retrieval results between standard query vector search and HyDE search.

    Calculates:
    - Overlap count and Jaccard similarity.
    - Chunks unique to standard retrieval vs HyDE.
    - Score differential on common chunks.
    """
    normal_ids = {c.chunk_id for c in normal_results}
    hyde_ids = {c.chunk_id for c in hyde_results}

    intersection = normal_ids & hyde_ids
    union = normal_ids | hyde_ids

    jaccard = len(intersection) / len(union) if union else 1.0
    normal_only = normal_ids - hyde_ids
    hyde_only = hyde_ids - normal_ids

    return {
        "normal_count": len(normal_results),
        "hyde_count": len(hyde_results),
        "overlap_count": len(intersection),
        "jaccard_similarity": round(jaccard, 4),
        "normal_only_ids": list(normal_only),
        "hyde_only_ids": list(hyde_only),
        "common_ids": list(intersection),
    }
