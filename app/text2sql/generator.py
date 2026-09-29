"""
Text-to-SQL Generator.

Translates natural language questions into safe PostgreSQL SELECT queries
using database schema context.
"""
import re
from typing import Tuple
from app.core.config import get_settings
from app.core.logging import logger
from app.text2sql.schema import get_schema_prompt_context
from app.rag.llm import get_chat_model

settings = get_settings()


def _extract_sql_from_markdown(text: str) -> str:
    """Extract SQL from markdown fenced blocks if present."""
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


async def generate_sql_query(question: str) -> Tuple[str, str]:
    """
    Generate a PostgreSQL SELECT query from a natural language question.

    Returns:
        (sql_query, explanation)
    """
    llm = get_chat_model()
    schema_context = get_schema_prompt_context()

    if llm:
        try:
            prompt = (
                f"You are a PostgreSQL expert database engineer.\n"
                f"Based on the database schema below, generate a safe, performant PostgreSQL SELECT query "
                f"that accurately answers the user's question.\n\n"
                f"{schema_context}\n\n"
                f"Rules:\n"
                f"1. Generate ONLY a single SELECT statement.\n"
                f"2. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or administrative queries.\n"
                f"3. Only use tables that exist in the schema context.\n"
                f"4. Include an explanation of what the query does.\n\n"
                f"Question: {question}\n\n"
                f"Format your response as:\n"
                f"EXPLANATION: <brief explanation>\n"
                f"SQL:\n"
                f"```sql\n<your query>\n```"
            )
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            explanation = "Generated SQL based on user question."
            if "EXPLANATION:" in content and "SQL:" in content:
                parts = content.split("SQL:")
                explanation = parts[0].replace("EXPLANATION:", "").strip()
                sql_part = parts[1].strip()
            else:
                sql_part = content

            raw_sql = _extract_sql_from_markdown(sql_part)
            if raw_sql:
                return raw_sql, explanation
        except Exception as e:
            logger.warning(f"LLM SQL generation failed: {e}. Using deterministic semantic generator.")

    # Semantic pattern generator for local development & testing
    q_lower = question.lower()

    # Pattern: Top N Kubernetes / Service incidents
    limit_match = re.search(r"\btop\s+(\d+)\b", q_lower)
    limit_num = int(limit_match.group(1)) if limit_match else 5

    if "incident" in q_lower:
        if "kubernetes" in q_lower or "k8s" in q_lower:
            sql = (
                f"SELECT id, title, service, severity, status, impact_summary, created_at "
                f"FROM incidents "
                f"WHERE LOWER(service) = 'kubernetes' OR LOWER(title) LIKE '%kubernetes%' "
                f"ORDER BY created_at DESC "
                f"LIMIT {limit_num}"
            )
            explanation = f"Retrieves the top {limit_num} most recent Kubernetes incidents with their details."
            return sql, explanation

        if "severity" in q_lower and ("count" in q_lower or "breakdown" in q_lower or "group" in q_lower):
            sql = (
                "SELECT severity, COUNT(*) AS incident_count "
                "FROM incidents "
                "GROUP BY severity "
                "ORDER BY incident_count DESC "
                "LIMIT 50"
            )
            explanation = "Aggregates incident counts grouped by severity level."
            return sql, explanation

        if "critical" in q_lower:
            sql = (
                f"SELECT id, title, service, severity, status, created_at "
                f"FROM incidents "
                f"WHERE severity = 'CRITICAL' "
                f"ORDER BY created_at DESC "
                f"LIMIT {limit_num}"
            )
            explanation = f"Lists the most recent CRITICAL severity incidents."
            return sql, explanation

        # General incidents
        sql = (
            f"SELECT id, title, service, severity, status, created_at "
            f"FROM incidents "
            f"ORDER BY created_at DESC "
            f"LIMIT {limit_num}"
        )
        explanation = f"Retrieves the top {limit_num} most recent production incidents."
        return sql, explanation

    # Pattern: Documents queries
    if "document" in q_lower or "file" in q_lower:
        if "count" in q_lower or "how many" in q_lower:
            sql = "SELECT COUNT(*) AS total_documents, SUM(total_chunks) AS total_chunks FROM documents"
            explanation = "Counts total number of ingested documents and chunks."
            return sql, explanation

        sql = (
            f"SELECT id, filename, file_type, file_size_bytes, total_chunks, status, created_at "
            f"FROM documents "
            f"ORDER BY created_at DESC "
            f"LIMIT {limit_num}"
        )
        explanation = f"Retrieves the {limit_num} most recently uploaded documents."
        return sql, explanation

    # Default fallback
    sql = (
        f"SELECT id, title, service, severity, status, created_at "
        f"FROM incidents "
        f"ORDER BY created_at DESC "
        f"LIMIT {limit_num}"
    )
    explanation = f"Default incident query for: '{question}'"
    return sql, explanation
