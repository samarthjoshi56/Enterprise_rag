import io
import uuid
from dataclasses import dataclass, field
from typing import List
import pypdf
from app.core.logging import logger


@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)


def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Parse PDF or TXT document content and extract page-by-page text & metadata.
    """
    document_id = str(uuid.uuid4())
    file_size = len(file_bytes)
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    if extension == "pdf":
        return _parse_pdf(file_bytes, filename, document_id, file_size)
    elif extension in ("txt", "text", "md"):
        return _parse_txt(file_bytes, filename, document_id, file_size)
    else:
        raise ValueError(f"Unsupported file format: .{extension}. Supported formats: .pdf, .txt")


def _parse_pdf(file_bytes: bytes, filename: str, document_id: str, file_size: int) -> ParsedDocument:
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    total_pages = len(reader.pages)
    pages: List[PageContent] = []

    for idx, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        extracted_text = extracted_text.strip()
        if extracted_text:
            pages.append(PageContent(page_number=idx, text=extracted_text))

    if not pages:
        logger.warning(f"No text extracted from PDF: {filename}")

    return ParsedDocument(
        document_id=document_id,
        filename=filename,
        file_type="pdf",
        file_size_bytes=file_size,
        total_pages=total_pages,
        pages=pages,
    )


def _parse_txt(file_bytes: bytes, filename: str, document_id: str, file_size: int) -> ParsedDocument:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="replace")

    text = text.strip()
    pages = [PageContent(page_number=1, text=text)] if text else []

    return ParsedDocument(
        document_id=document_id,
        filename=filename,
        file_type="txt",
        file_size_bytes=file_size,
        total_pages=1,
        pages=pages,
    )
