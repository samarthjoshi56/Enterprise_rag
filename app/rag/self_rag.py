"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation (Asai et al., 2023).

Implements reflection tokens and decision mechanisms:
1. [Retrieve]: Decides whether external retrieval is required for the input query.
2. [IsREL]: Assesses whether each retrieved chunk is relevant to the query.
3. [IsSUP]: Validates whether generated statements/claims are supported by the context.
4. [IsUSE]: Evaluates overall utility of the retrieved context for answering the query.
"""
import re
from typing import List, Dict, Any, Tuple
from app.core.config import get_settings
from app.core.logging import logger
from app.search.models import ChunkResult
from app.search.reranker import get_reranker
from app.rag.llm import evaluate_hallucination

settings = get_settings()

# Reflection token constants
TOKEN_RETRIEVE_YES = "[Retrieve]"
TOKEN_RETRIEVE_NO = "[No Retrieve]"
TOKEN_IS_REL_YES = "[Relevant]"
TOKEN_IS_REL_NO = "[Irrelevant]"
TOKEN_IS_SUP_FULLY = "[Fully Supported]"
TOKEN_IS_SUP_PARTIALLY = "[Partially Supported]"
TOKEN_IS_SUP_NO = "[No Support]"


def should_retrieve(query: str) -> Tuple[bool, str]:
    """
    Self-RAG [Retrieve] token decision:
    Determines whether the user query requires external knowledge retrieval.

    Queries that don't need retrieval:
    - Conversational greetings ("hello", "good morning", "thank you")
    - Simple chit-chat / meta questions ("who are you", "what can you do")
    - Direct arithmetic ("2 + 2")

    Returns:
        (should_retrieve_bool, token_string)
    """
    clean = query.strip().lower()

    # Conversational / greeting patterns
    greeting_patterns = [
        r"^(hi|hello|hey|howdy|greetings|good morning|good afternoon|good evening)\b",
        r"^(thanks|thank you|thx)\b",
        r"^(who are you|what can you do|help)\b",
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, clean):
            logger.info(f"Self-RAG [Retrieve] decision: NO (conversational: '{query}')")
            return False, TOKEN_RETRIEVE_NO

    # Math expressions without context
    if re.match(r"^[\d\s\+\-\*\/\(\)\=\.]+$", clean):
        logger.info(f"Self-RAG [Retrieve] decision: NO (arithmetic: '{query}')")
        return False, TOKEN_RETRIEVE_NO

    logger.info(f"Self-RAG [Retrieve] decision: YES for query: '{query[:60]}'")
    return True, TOKEN_RETRIEVE_YES


def evaluate_chunk_relevance(
    query: str,
    chunk: ChunkResult,
    threshold: float = None,
) -> Tuple[bool, str, float]:
    """
    Self-RAG [IsREL] token decision:
    Evaluates whether an individual retrieved chunk is relevant to the query.

    Returns:
        (is_relevant, token_str, score)
    """
    if threshold is None:
        threshold = settings.CRAG_RELEVANCE_THRESHOLD

    # Use cross-encoder score if available, otherwise compute it
    score = chunk.rerank_score
    if score is None:
        reranker = get_reranker()
        score = float(reranker.predict([(query, chunk.text)])[0])
        chunk.rerank_score = round(score, 6)

    is_rel = score >= threshold
    token = TOKEN_IS_REL_YES if is_rel else TOKEN_IS_REL_NO
    return is_rel, token, score


def filter_relevant_chunks(
    query: str,
    chunks: List[ChunkResult],
    threshold: float = None,
) -> Tuple[List[ChunkResult], List[Dict[str, Any]]]:
    """
    Self-RAG [IsREL] batch assessment:
    Filters chunks and returns list of relevant chunks alongside audit reflections.
    """
    if threshold is None:
        threshold = settings.CRAG_RELEVANCE_THRESHOLD

    relevant: List[ChunkResult] = []
    reflections: List[Dict[str, Any]] = []

    for c in chunks:
        is_rel, token, score = evaluate_chunk_relevance(query, c, threshold=threshold)
        reflections.append({
            "chunk_id": c.chunk_id,
            "filename": c.filename,
            "score": score,
            "token": token,
            "is_relevant": is_rel,
        })
        if is_rel:
            relevant.append(c)

    logger.info(
        f"Self-RAG [IsREL]: {len(relevant)}/{len(chunks)} chunks classified as relevant "
        f"(threshold={threshold})"
    )
    return relevant, reflections


def decide_more_retrieval_needed(
    query: str,
    relevant_chunks: List[ChunkResult],
    min_required_chunks: int = 1,
) -> Tuple[bool, str]:
    """
    Self-RAG: Decide whether more retrieval is required.
    If no chunks are relevant, or context is insufficient, returns True.
    """
    if len(relevant_chunks) < min_required_chunks:
        reason = f"Insufficient relevant chunks ({len(relevant_chunks)} < {min_required_chunks})"
        logger.info(f"Self-RAG More Retrieval: YES ({reason})")
        return True, reason

    logger.info("Self-RAG More Retrieval: NO (retrieval is sufficient)")
    return False, "Sufficient relevant chunks available"


async def validate_answer_support(
    context_chunks: List[ChunkResult],
    statement: str,
) -> Tuple[bool, str]:
    """
    Self-RAG [IsSUP] validation:
    Validates whether generated information is supported by the retrieved context
    (hallucination verification).
    """
    if not context_chunks:
        return False, TOKEN_IS_SUP_NO

    full_context = "\n---\n".join([c.text for c in context_chunks])
    supported = await evaluate_hallucination(full_context, statement)

    token = TOKEN_IS_SUP_FULLY if supported else TOKEN_IS_SUP_NO
    logger.info(f"Self-RAG [IsSUP] check: {token} for statement='{statement[:60]}'")
    return supported, token
