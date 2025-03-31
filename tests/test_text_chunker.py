"""
Tests for the text chunking functionality.
"""

import pytest
from rag.processing.text_chunker import TextChunker


def test_basic_chunking():
    """Test basic text chunking with default settings."""
    chunker = TextChunker()
    text = "This is a test sentence. This is another test sentence."
    chunks = chunker.chunk_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_size_and_overlap():
    """Test if chunks respect size and overlap parameters."""
    chunk_size = 20
    chunk_overlap = 5
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Create a text that should definitely be split
    text = "a " * 50  # This should create a text of about 100 characters
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 1  # Should be split into multiple chunks
    assert all(len(chunk) <= chunk_size for chunk in chunks)  # Each chunk should respect max size


def test_metadata_handling():
    """Test chunking with metadata."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "This is a test paragraph. It contains multiple sentences for testing."
    source_info = {
        "source_type": "test",
        "document_id": "test_001"
    }
    
    chunks = chunker.chunk_text_with_metadata(text, source_info)
    
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    
    # Check metadata contents
    for i, chunk in enumerate(chunks, 1):
        assert chunk["metadata"]["chunk_id"] == i
        assert chunk["metadata"]["source_type"] == "test"
        assert chunk["metadata"]["document_id"] == "test_001"
        assert "chunk_length" in chunk["metadata"]


def test_separator_handling():
    """Test if text is split at appropriate separators."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    # Test with different separators
    text = "First paragraph.\n\nSecond paragraph.\nThird line. Fourth sentence!"
    chunks = chunker.chunk_text(text)
    
    # The text should be split at paragraph breaks first
    assert any("First paragraph" in chunk for chunk in chunks)
    assert any("Second paragraph" in chunk for chunk in chunks)
    assert any("Third line" in chunk for chunk in chunks)


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    chunker = TextChunker()
    
    # Test empty string
    assert chunker.chunk_text("") == []
    
    # Test very short text (shorter than chunk_size)
    short_text = "Short text."
    chunks = chunker.chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    # Test with metadata for empty text
    empty_with_metadata = chunker.chunk_text_with_metadata("", {"source": "empty"})
    assert len(empty_with_metadata) == 0


def test_chunk_position_metadata():
    """Test if chunk position metadata is correct."""
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "This is a longer text that should be split into multiple chunks."
    
    chunks = chunker.chunk_text_with_metadata(text)
    
    assert chunks[0]["metadata"]["chunk_position"] == "start"
    assert chunks[-1]["metadata"]["chunk_position"] == "end"
    
    if len(chunks) > 2:
        assert all(chunk["metadata"]["chunk_position"] == "middle" 
                  for chunk in chunks[1:-1]) 