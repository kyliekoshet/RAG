"""
Tests for the PDF Parser module.
"""

import os
import pytest
from pathlib import Path
from rag.processing.pdf_parser import PDFParser

# Sample PDF path
SAMPLE_PDF_PATH = Path("/Users/kyliekoshet/Desktop/MyProjects/RAG/data/2003-03-04-4.pdf")

def test_pdf_parser_initialization():
    """Test PDFParser initialization with valid and invalid paths."""
    # Test with valid PDF
    parser = PDFParser(str(SAMPLE_PDF_PATH))
    assert parser.pdf_path == SAMPLE_PDF_PATH
    
    # Test with non-existent PDF
    with pytest.raises(FileNotFoundError):
        PDFParser("non_existent.pdf")

def test_pdf_text_extraction():
    """Test text extraction from the sample PDF."""
    parser = PDFParser(str(SAMPLE_PDF_PATH))
    extracted_text = parser.extract_text()
    
    # Basic validation of extraction results
    assert isinstance(extracted_text, list)
    assert len(extracted_text) > 0
    assert all(isinstance(text, str) for text in extracted_text)
    assert all(len(text.strip()) > 0 for text in extracted_text)
    
    # Check if we got some expected content from the PDF
    content = " ".join(extracted_text)
    assert "Phase I" in content
    assert "Cannabis Based Medicine Extract" in content

def test_text_cleaning():
    """Test the text cleaning functionality."""
    parser = PDFParser(str(SAMPLE_PDF_PATH))
    test_texts = [
        "  Multiple    spaces   ",
        "\n\nNew lines\n\n",
        "",  # Empty string
        "   ",  # Only whitespace
        "Normal text"
    ]
    
    cleaned_texts = parser._clean_text_elements(test_texts)
    
    # Verify cleaning results
    assert all(text == text.strip() for text in cleaned_texts)  # All texts are stripped
    assert all(" " * 2 not in text for text in cleaned_texts)  # No double spaces
    assert all(len(text) > 0 for text in cleaned_texts)  # No empty strings
    assert "Multiple spaces" in cleaned_texts
    assert "Normal text" in cleaned_texts
    assert "New lines" in cleaned_texts

def test_extraction_with_titles():
    """Test text extraction with and without titles."""
    parser = PDFParser(str(SAMPLE_PDF_PATH))
    
    # Test with titles included
    with_titles = parser.extract_text(include_titles=True)
    assert len(with_titles) > 0
    
    # Test without titles
    without_titles = parser.extract_text(include_titles=False)
    assert len(without_titles) > 0
    
    # Both methods should extract meaningful text
    assert any("Phase I" in text for text in with_titles)
    assert any("Phase I" in text for text in without_titles) 