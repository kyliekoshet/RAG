"""
Configuration module for RAG system.

This module contains all the configuration settings for the RAG system,
including embedding model settings, vector store settings, and defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
import os
from dotenv import load_dotenv

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: Literal["openai", "huggingface"] = "huggingface"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Default for all-MiniLM-L6-v2
    device: str = "cpu"

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector store."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "hybrid_vectors"
    prefer_grpc: bool = True
    timeout: float = 5.0

@dataclass
class FaissCacheConfig:
    """Configuration for FAISS cache."""
    cache_size: int = 10000  # Maximum vectors in cache
    index_type: Literal["l2", "cosine"] = "cosine"
    access_threshold: int = 5  # Times accessed before caching
    batch_size: int = 1000

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: Literal["faiss", "qdrant", "hybrid"] = "hybrid"
    save_dir: Optional[str] = "./vectors"
    # FAISS-specific settings
    faiss: FaissCacheConfig = field(default_factory=FaissCacheConfig)
    # Qdrant-specific settings
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

@dataclass
class RAGConfig:
    """Main configuration for RAG system."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    
    def __post_init__(self):
        """Load environment variables and validate configuration."""
        load_dotenv()
        
        # Override with environment variables if they exist
        if os.getenv("EMBEDDING_PROVIDER"):
            self.embedding.provider = os.getenv("EMBEDDING_PROVIDER")
        if os.getenv("EMBEDDING_MODEL"):
            self.embedding.model_name = os.getenv("EMBEDDING_MODEL")
        if os.getenv("VECTOR_STORE_DIR"):
            self.vector_store.save_dir = os.getenv("VECTOR_STORE_DIR")
        if os.getenv("VECTOR_STORE_TYPE"):
            self.vector_store.store_type = os.getenv("VECTOR_STORE_TYPE")
        if os.getenv("QDRANT_HOST"):
            self.vector_store.qdrant.host = os.getenv("QDRANT_HOST")
        if os.getenv("QDRANT_PORT"):
            self.vector_store.qdrant.port = int(os.getenv("QDRANT_PORT"))

# Default configuration instance
default_config = RAGConfig()

# Example configuration with hybrid store
hybrid_config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="huggingface",
        model_name="all-MiniLM-L6-v2"
    ),
    vector_store=VectorStoreConfig(
        store_type="hybrid",
        save_dir="./vectors",
        faiss=FaissCacheConfig(
            cache_size=10000,
            index_type="cosine",
            access_threshold=5
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="hybrid_vectors"
        )
    )
) 