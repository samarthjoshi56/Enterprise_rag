import uuid
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings
from app.ingestion.loader import ParsedDocument

settings = get_settings()


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[DocumentChunk]:
    """
    Split a ParsedDocument into granular chunks with metadata.
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[DocumentChunk] = []
    global_chunk_index = 0

    for page in parsed_doc.pages:
        if not page.text.strip():
            continue

        page_splits = splitter.split_text(page.text)
        for split_text in page_splits:
            split_text = split_text.strip()
            if not split_text:
                continue

            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=parsed_doc.document_id,
                filename=parsed_doc.filename,
                page_number=page.page_number,
                chunk_index=global_chunk_index,
                text=split_text,
                character_count=len(split_text),
            )
            chunks.append(chunk)
            global_chunk_index += 1

    return chunks
