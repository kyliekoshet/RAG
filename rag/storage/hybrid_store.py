"""
Hybrid vector store combining FAISS and Qdrant.

This store uses:
- FAISS for fast in-memory search of frequently accessed vectors
- Qdrant for persistent storage and advanced filtering
- Intelligent caching strategy to keep hot vectors in FAISS
"""

from typing import List, Dict, Optional, Union, Set
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..core.models.vector_store_models import VectorMetadata
from .base_store import BaseVectorStore
from .faiss_store import FaissStore
from .qdrant_store import QdrantStore
from ..core.config import RAGConfig, default_config
from ..embeddings.providers import TextEmbedder

class HybridStore(BaseVectorStore):
    """Hybrid vector store using both FAISS and Qdrant."""
    
    def __init__(
        self,
        dimension: int,
        config: Optional[RAGConfig] = None,
        embedder: Optional[TextEmbedder] = None,
        collection_name: str = "hybrid_vectors",
        cache_size: int = 10000  # Max vectors to keep in FAISS
    ):
        """Initialize the hybrid store.
        
        Args:
            dimension: Vector dimension
            config: RAG configuration
            embedder: Text embedder instance
            collection_name: Qdrant collection name
            cache_size: Maximum vectors to keep in FAISS cache
        """
        super().__init__(dimension=dimension, config=config)
        self.embedder = embedder
        self.collection_name = collection_name
        self.cache_size = cache_size
        
        # Initialize both stores
        self.faiss_store = FaissStore(
            dimension=dimension,
            config=self._config,
            embedder=embedder
        )
        
        self.qdrant_store = QdrantStore(
            dimension=dimension,
            config=self._config,
            collection_name=collection_name
        )
        
        # Cache tracking
        self.access_counts: Dict[int, int] = {}  # Vector ID -> access count
        self.last_accessed: Dict[int, datetime] = {}  # Vector ID -> last access time

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add vectors to both stores."""
        # Add to Qdrant first (persistent storage)
        qdrant_ids = self.qdrant_store.add_vectors(vectors, metadata)
        
        # Add to FAISS cache if space available
        if self.faiss_store.index.ntotal < self.cache_size:
            self.faiss_store.add_vectors(vectors, metadata)
            
            # Initialize access tracking
            for vid in qdrant_ids:
                self.access_counts[vid] = 0
                self.last_accessed[vid] = datetime.now()
        
        return qdrant_ids

    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        use_cache: bool = True
    ) -> List[VectorMetadata]:
        """Search for similar vectors.
        
        Args:
            vector: Query vector
            k: Number of results
            filter: Metadata filter
            use_cache: Whether to use FAISS cache
        """
        results = []
        
        # Try FAISS cache first if enabled and no filter
        if use_cache and not filter and self.faiss_store.index.ntotal > 0:
            results = self.faiss_store.search_vector(vector, k=k)
            
            # Update access stats
            for result in results:
                vid = result.vector_id
                self.access_counts[vid] = self.access_counts.get(vid, 0) + 1
                self.last_accessed[vid] = datetime.now()
                
        # Fall back to Qdrant if needed
        if not results:
            results = self.qdrant_store.search_vector(vector, k=k, filter=filter)
            
            # Consider caching frequently accessed vectors
            self._update_cache(results)
            
        return results

    def _update_cache(self, results: List[VectorMetadata]):
        """Update FAISS cache based on access patterns."""
        for result in results:
            vid = result.vector_id
            count = self.access_counts.get(vid, 0)
            
            # Add to cache if frequently accessed
            if count > 5 and self.faiss_store.index.ntotal < self.cache_size:
                # Get vector from Qdrant
                vector = self.qdrant_store.get_vector(vid)
                if vector is not None:
                    # Add to FAISS cache
                    self.faiss_store.add_vectors(
                        vectors=vector.reshape(1, -1),
                        metadata=[result.to_dict()]
                    )

    def get_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get statistics about both stores."""
        faiss_stats = self.faiss_store.get_stats()
        qdrant_stats = self.qdrant_store.get_stats()
        
        return {
            "total_vectors": qdrant_stats["total_vectors"],
            "cached_vectors": faiss_stats["total_vectors"],
            "cache_hit_ratio": len(self.access_counts) / qdrant_stats["total_vectors"],
            "dimension": self.dimension,
            "collection": self.collection_name
        }

    def save(self, save_dir: Optional[str] = None) -> None:
        """Save both stores."""
        # Only need to save Qdrant config since it's already persistent
        # and FAISS is just a cache
        self.qdrant_store.save(save_dir)
        
    @classmethod
    def load(cls, load_dir: str, **kwargs) -> "HybridStore":
        """Load the hybrid store."""
        # Create new instance
        instance = cls(
            dimension=kwargs.get("dimension", 384),
            config=kwargs.get("config"),
            embedder=kwargs.get("embedder"),
            collection_name=kwargs.get("collection_name", "hybrid_vectors")
        )
        
        # Load Qdrant store
        instance.qdrant_store = QdrantStore.load(load_dir)
        
        # FAISS cache will be rebuilt as needed
        return instance 