from app.ingestion.loader import ParsedDocument, PageContent
from app.ingestion.chunker import chunk_document


def test_chunk_document_basic():
    """Test text splitting into configurable chunks."""
    long_text = "Word " * 300  # ~1500 chars
    parsed_doc = ParsedDocument(
        document_id="doc-123",
        filename="test.txt",
        file_type="txt",
        file_size_bytes=len(long_text),
        total_pages=1,
        pages=[PageContent(page_number=1, text=long_text)],
    )

    chunks = chunk_document(parsed_doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc-123"
    assert chunks[0].filename == "test.txt"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert len(chunks[0].text) <= 500
    assert chunks[1].chunk_index == 1
