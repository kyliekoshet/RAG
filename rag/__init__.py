"""
RAG (Retrieval-Augmented Generation) package.

This package provides tools for building RAG systems, including:
- Document parsing and chunking
- Vector storage and retrieval
- Embedding generation
- Configuration management
"""

from .core.config import RAGConfig, default_config, hybrid_config
from .embeddings.providers import TextEmbedder
from .storage.hybrid_store import HybridStore
from .storage.faiss_store import FaissStore
from .storage.qdrant_store import QdrantStore
from .processing.pdf_parser import PDFParser
from .processing.text_chunker import TextChunker

__version__ = "0.1.0"

__all__ = [
    'RAGConfig', 
    'default_config', 
    'hybrid_config', 
    'TextEmbedder', 
    'HybridStore', 
    'FaissStore', 
    'QdrantStore',
    'PDFParser',
    'TextChunker'
] 
