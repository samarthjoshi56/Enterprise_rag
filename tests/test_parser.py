import io
import pytest
from app.ingestion.loader import parse_document


def test_parse_txt_document():
    """Test parsing a plain text file."""
    content = "Enterprise RAG System\n\nThis is a test text document for Phase 2 document ingestion."
    file_bytes = content.encode("utf-8")
    
    parsed = parse_document(file_bytes, "sample.txt")
    
    assert parsed.filename == "sample.txt"
    assert parsed.file_type == "txt"
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_number == 1
    assert "Enterprise RAG" in parsed.pages[0].text


def test_parse_pdf_document():
    """Test parsing a PDF document generated dynamically in memory."""
    from reportlab.pdfgen import canvas
    
    pdf_bytes_io = io.BytesIO()
    c = canvas.Canvas(pdf_bytes_io)
    c.drawString(100, 750, "Sample PDF text for unit test.")
    c.showPage()
    c.save()
    
    pdf_bytes = pdf_bytes_io.getvalue()
    parsed = parse_document(pdf_bytes, "sample.pdf")
    
    assert parsed.filename == "sample.pdf"
    assert parsed.file_type == "pdf"
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
    assert "Sample PDF text" in parsed.pages[0].text
