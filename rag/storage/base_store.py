"""
Base class for vector stores in the RAG system.

This module defines the interface that all vector store implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import numpy as np
from ..core.models.vector_store_models import VectorMetadata
from ..core.config import RAGConfig, default_config

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(
        self,
        dimension: int,
        config: Optional[RAGConfig] = None,
        **kwargs
    ):
        """Initialize the vector store.
        
        Args:b
            dimension: Dimension of vectors to store
            config: RAG configuration
            **kwargs: Additional implementation-specific arguments
        """
        self._dimension = dimension
        self._config = config or default_config
        
    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in the store."""
        return self._dimension
        
    @property
    def config(self) -> Optional[RAGConfig]:
        """Get the store configuration."""
        return self._config
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[VectorMetadata]
    ) -> List[int]:
        """Add vectors to the store.
        
        Args:
            vectors: Array of vectors to add
            metadata: List of metadata for each vector
            
        Returns:
            List of assigned IDs
        """
        pass
    
    @abstractmethod
    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None
    ) -> List[VectorMetadata]:
        """Search for similar vectors.
        
        Args:
            vector: Query vector
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of metadata for similar vectors
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Union[int, str]]:
        """Get statistics about the store.
        
        Returns:
            Dictionary with stats
        """
        pass 