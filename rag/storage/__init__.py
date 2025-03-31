"""
Vector storage module for RAG system.

This module provides different vector store implementations and a unified interface.
"""

from typing import Optional
from .base_store import BaseVectorStore
from .faiss_store import FaissStore
from .qdrant_store import QdrantStore
from .hybrid_store import HybridStore
from ..core.config import RAGConfig, default_config
from ..embeddings.providers import TextEmbedder

def create_vector_store(
    store_type: str = "faiss",
    dimension: Optional[int] = None,
    config: Optional[RAGConfig] = None,
    embedder: Optional[TextEmbedder] = None,
    **kwargs
) -> BaseVectorStore:
    """
    Create a vector store of the specified type.
    
    Args:
        store_type: Type of store to create ('faiss' or 'qdrant')
        dimension: Dimension of vectors to store
        config: RAG configuration
        embedder: TextEmbedder instance for automatic model tracking
        **kwargs: Additional arguments passed to the store constructor
        
    Returns:
        Initialized vector store
        
    Raises:
        ValueError: If store_type is invalid or required args are missing
    """
    config = config or default_config
    
    # Determine dimension
    if dimension is None:
        if embedder is not None:
            dimension = embedder.dimension
        else:
            dimension = config.embedding.dimension
            
    if dimension <= 0:
        raise ValueError("Dimension must be positive")
    
    # Create store based on type
    if store_type == "faiss":
        return FaissStore(
            dimension=dimension,
            config=config,
            embedder=embedder,
            **kwargs
        )
    elif store_type == "qdrant":
        return QdrantStore(
            dimension=dimension,
            config=config,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")

__all__ = ['BaseVectorStore', 'FaissStore', 'QdrantStore', 'HybridStore', 'create_vector_store'] 