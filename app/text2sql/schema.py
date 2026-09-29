"""
Database schema context provider for Text2SQL.

Defines the allowed tables and generates structured schema representations
for LLM prompts and validation checks.
"""
from typing import Dict, Any, List

ALLOWED_TABLES = {"documents", "document_chunks", "incidents"}

# Strict schema definitions with column metadata and usage hints
SCHEMA_METADATA: Dict[str, Dict[str, Any]] = {
    "incidents": {
        "description": "Production engineering and DevOps incidents (e.g. Kubernetes outages, memory spikes, service errors)",
        "columns": {
            "id": "VARCHAR(36) PRIMARY KEY",
            "title": "VARCHAR(255) NOT NULL - Short summary of the incident",
            "service": "VARCHAR(100) NOT NULL - Affected service name (e.g. 'kubernetes', 'postgresql', 'redis')",
            "severity": "VARCHAR(50) NOT NULL - Severity level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')",
            "status": "VARCHAR(50) NOT NULL - Incident status ('RESOLVED', 'INVESTIGATING', 'OPEN')",
            "impact_summary": "TEXT - Detailed technical impact and resolution details",
            "created_at": "TIMESTAMP WITH TIME ZONE - Incident start time",
            "resolved_at": "TIMESTAMP WITH TIME ZONE - Incident resolution time",
        },
    },
    "documents": {
        "description": "Metadata for all ingested documents in the enterprise repository",
        "columns": {
            "id": "VARCHAR(36) PRIMARY KEY",
            "filename": "VARCHAR(255) NOT NULL - Original filename (e.g. 'architecture.pdf')",
            "file_type": "VARCHAR(50) NOT NULL - File extension ('pdf', 'txt')",
            "file_size_bytes": "INTEGER - File size in bytes",
            "total_chunks": "INTEGER - Total number of chunks generated",
            "total_pages": "INTEGER - Total page count",
            "status": "VARCHAR(50) - Ingestion status ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')",
            "error_message": "TEXT - Processing error if failed",
            "created_at": "TIMESTAMP WITH TIME ZONE - Ingestion timestamp",
            "updated_at": "TIMESTAMP WITH TIME ZONE - Last update timestamp",
        },
    },
    "document_chunks": {
        "description": "Individual chunk segments extracted from documents for vector and keyword search",
        "columns": {
            "id": "VARCHAR(36) PRIMARY KEY",
            "document_id": "VARCHAR(36) FOREIGN KEY REFERENCES documents(id)",
            "chunk_index": "INTEGER - Zero-indexed order within document",
            "page_number": "INTEGER - Original page number",
            "qdrant_point_id": "VARCHAR(36) - Corresponding Qdrant vector point ID",
            "character_count": "INTEGER - Length of chunk text",
            "chunk_text": "TEXT - Raw text of the chunk",
            "created_at": "TIMESTAMP WITH TIME ZONE - Chunk creation timestamp",
        },
    },
}


def get_allowed_tables() -> List[str]:
    """Return list of table names allowed in Text2SQL queries."""
    return sorted(list(ALLOWED_TABLES))


def get_schema_prompt_context() -> str:
    """Format schema metadata as a readable DDL-style reference for the LLM."""
    lines = ["PostgreSQL Database Schema Context (Allowed tables only):", ""]
    for table_name, meta in SCHEMA_METADATA.items():
        lines.append(f"Table: {table_name}")
        lines.append(f"Description: {meta['description']}")
        lines.append("Columns:")
        for col, col_type in meta["columns"].items():
            lines.append(f"  - {col}: {col_type}")
        lines.append("")

    return "\n".join(lines)
