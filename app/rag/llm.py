"""
LLM abstraction layer for the entire RAG platform.

Provider priority:
1. Google Gemini (when GOOGLE_API_KEY is set and LLM_PROVIDER == "google")
2. OpenAI (when OPENAI_API_KEY is set and LLM_PROVIDER == "openai")
3. Offline / local fallback (heuristic text generators — no external API needed)

Used by: HyDE, CRAG, Self-RAG, Text2SQL, and the Chat answer generator.
"""
import re
from typing import Optional, List
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

_chat_model = None
_chat_model_initialized = False


def get_chat_model():
    """
    Return a LangChain chat model (Gemini or OpenAI), cached as a singleton.
    Returns None if no API key is configured.
    """
    global _chat_model, _chat_model_initialized
    if _chat_model_initialized:
        return _chat_model

    _chat_model_initialized = True

    # --- Google Gemini ---
    if settings.LLM_PROVIDER == "google" and settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _chat_model = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL_NAME,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.0,
                convert_system_message_to_human=True,
            )
            logger.info(f"LLM initialized: Google Gemini ({settings.LLM_MODEL_NAME})")
            return _chat_model
        except Exception as e:
            logger.warning(f"Could not initialize Google Gemini: {e}. Trying OpenAI fallback.")

    # --- OpenAI ---
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
        try:
            from langchain_openai import ChatOpenAI
            _chat_model = ChatOpenAI(
                model=settings.LLM_MODEL_NAME if settings.LLM_PROVIDER == "openai" else "gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
                temperature=0.0,
            )
            logger.info(f"LLM initialized: OpenAI ({_chat_model.model_name})")
            return _chat_model
        except Exception as e:
            logger.warning(f"Could not initialize ChatOpenAI: {e}. Using local fallback.")

    logger.warning("No LLM API key configured. Using rule-based local fallback for all LLM calls.")
    return None


# ─── Answer Generation ──────────────────────────────────────────────────────


async def generate_rag_answer(query: str, context_chunks: List[dict]) -> str:
    """
    Generate a grounded answer using the LLM with retrieved context.
    This is the main Chat answer generator.

    Args:
        query: User's question.
        context_chunks: List of dicts with at least "text", "filename", "page_number".

    Returns:
        A natural-language answer grounded in the provided context.
    """
    llm = get_chat_model()

    # Build context block
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        page = c.get("page_number") or "?"
        context_parts.append(
            f"[Source {i}: {c.get('filename', 'unknown')} (page {page})]\n{c['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(no context)"

    if llm:
        try:
            prompt = (
                "You are a knowledgeable enterprise assistant. Answer the user's question "
                "based ONLY on the provided context. If the context does not contain enough "
                "information, say so clearly. Cite your sources using [Source N] references.\n\n"
                f"### Context\n\n{context_block}\n\n"
                f"### Question\n\n{query}\n\n"
                "### Answer\n\n"
            )
            response = await llm.ainvoke(prompt)
            answer = response.content.strip()
            if answer:
                return answer
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}. Using fallback.")

    # Fallback: structured summary from chunks
    if not context_chunks:
        return (
            "I couldn't find relevant information in your documents for that query. "
            "Try uploading more documents or rephrasing your question."
        )

    parts = []
    for i, c in enumerate(context_chunks[:3], 1):
        text = c["text"][:400] + ("…" if len(c["text"]) > 400 else "")
        parts.append(f"**{i}. {c.get('filename', '?')}** (p.{c.get('page_number', '?')}):\n{text}")
    return (
        f"Based on your documents, here are the most relevant findings for "
        f"**\"{query}\"**:\n\n" + "\n\n".join(parts)
    )


# ─── HyDE ────────────────────────────────────────────────────────────────────


async def generate_hypothetical_document(query: str) -> str:
    """
    HyDE: Generate a hypothetical document passage that answers the query.
    This passage is then embedded to perform semantic vector search.
    """
    llm = get_chat_model()
    if llm:
        try:
            prompt = (
                f"You are a technical documentation writer. Write a comprehensive, factual "
                f"passage (2-3 paragraphs) answering the following query. "
                f"Include specific technical details, configurations, and terminology.\n\n"
                f"Query: {query}\n\n"
                f"Passage:"
            )
            response = await llm.ainvoke(prompt)
            hypo_text = response.content.strip()
            if hypo_text:
                return hypo_text
        except Exception as e:
            logger.warning(f"LLM HyDE generation failed: {e}. Falling back to rule-based generator.")

    # Rule-based fallback
    clean_query = query.strip().rstrip("?").strip()
    return (
        f"Technical documentation and architectural specification regarding {clean_query}. "
        f"This document outlines the standard configurations, operational requirements, "
        f"prerequisites, deployment steps, and best practices relevant to {clean_query}. "
        f"Implementation details include component architecture, service dependencies, "
        f"monitoring metrics, and resource specifications."
    )


# ─── CRAG query rewriter ────────────────────────────────────────────────────


async def rewrite_query(query: str, reason: str = "") -> str:
    """
    CRAG: Rewrite or expand a query when retrieval quality is poor or ambiguous.
    """
    llm = get_chat_model()
    if llm:
        try:
            prompt = (
                f"You are an expert search query engineer. The previous search for '{query}' "
                f"yielded insufficient results (Reason: {reason or 'low relevance'}). "
                f"Rewrite this query into a clearer, more effective search query focusing on "
                f"core technical keywords and synonyms. Respond ONLY with the rewritten query text.\n\n"
                f"Rewritten Query:"
            )
            response = await llm.ainvoke(prompt)
            rewritten = response.content.strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except Exception as e:
            logger.warning(f"LLM query rewrite failed: {e}. Falling back to rule-based rewriter.")

    # Rule-based fallback
    stop_words = {
        "what", "is", "are", "how", "to", "the", "a", "an", "in", "on", "for",
        "of", "do", "does", "can", "you", "tell", "me", "about", "please", "why"
    }
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
    meaningful = [t for t in tokens if t.lower() not in stop_words]

    if meaningful:
        return " ".join(meaningful)
    return query


# ─── Self-RAG hallucination checker ─────────────────────────────────────────


async def evaluate_hallucination(context: str, claim: str) -> bool:
    """
    Self-RAG [IsSUP]: Check whether a statement or claim is supported by the context.
    Returns True if supported, False if unsupported/hallucinated.
    """
    llm = get_chat_model()
    if llm:
        try:
            prompt = (
                f"Given the following Context:\n{context}\n\n"
                f"Is the Claim below fully supported by the Context? Answer YES or NO.\n\n"
                f"Claim: {claim}\nAnswer:"
            )
            response = await llm.ainvoke(prompt)
            return "yes" in response.content.lower()
        except Exception as e:
            logger.warning(f"LLM hallucination check failed: {e}. Falling back to token overlap.")

    # Fallback: check significant token overlap
    claim_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{4,}\b", claim.lower()))
    if not claim_tokens:
        return True
    context_lower = context.lower()
    matches = sum(1 for t in claim_tokens if t in context_lower)
    return (matches / len(claim_tokens)) >= 0.5
