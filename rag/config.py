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
class VectorStoreConfig:
    """Configuration for vector store."""
    index_type: Literal["l2", "cosine"] = "cosine"
    save_dir: Optional[str] = None
    batch_size: int = 1000

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

# Default configuration instance
default_config = RAGConfig()

# All settings in one place
config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="huggingface",
        model_name="all-MiniLM-L6-v2"
    ),
    vector_store=VectorStoreConfig(
        index_type="cosine",
        save_dir="./vectors"
    )
) 