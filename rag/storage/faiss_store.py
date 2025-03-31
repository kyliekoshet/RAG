"""
FAISS-based vector store for RAG system.

This module handles storing and retrieving embeddings using FAISS.
Storage Details:
    - Vectors are stored in FAISS index (in-memory by default)
    - Metadata is stored in a Python dictionary (in-memory)
    - Each vector gets a unique ID that links it to its metadata
    - Can be persisted to disk (save/load functionality)
    - Supports efficient batch operations
"""

from typing import List, Dict, Optional, Union, Set
import numpy as np
import faiss
import os
import json
import psutil
from datetime import datetime
from ..core.models.vector_store_models import VectorMetadata
from .base_store import BaseVectorStore
from ..core.config import RAGConfig, default_config
from ..embeddings.providers import TextEmbedder

class FaissStore(BaseVectorStore):
    """Vector store implementation using FAISS."""
    
    def __init__(
        self,
        dimension: int,
        config: Optional[RAGConfig] = None,
        embedder: Optional[TextEmbedder] = None,
        save_dir: Optional[str] = None
    ):
        """Initialize the FAISS store.
        
        Args:
            dimension: Dimension of vectors to store
            config: RAG configuration
            embedder: TextEmbedder instance for automatic model tracking
            save_dir: Directory to save/load the index and metadata
        """
        super().__init__(dimension=dimension, config=config)
        self.embedder = embedder
        self.save_dir = save_dir or self.config.vector_store.save_dir
            
        # Get index type from config
        self.index_type = self.config.vector_store.faiss.index_type
        
        # Initialize FAISS index
        if self.index_type == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        # Initialize metadata storage
        self.metadata: Dict[int, VectorMetadata] = {}
        self.next_id: int = 0
        self.initial_memory = self._get_memory_usage()
        
        # Create default metadata
        self.default_metadata = VectorMetadata(
            text="",  # Empty text for default metadata
            source="faiss_store",  # Source indicating this is from FAISS store
            embedding_model=embedder.model_name if embedder else "unknown"
        )

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add multiple vectors to the store in batches."""
        if len(vectors.shape) != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors must have shape (n, {self.dimension})")
            
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError("Metadata length must match number of vectors")
            
        # Process in batches
        batch_size = self.config.vector_store.faiss.batch_size
        n_vectors = len(vectors)
        assigned_ids = []
        
        for i in range(0, n_vectors, batch_size):
            batch_end = min(i + batch_size, n_vectors)
            batch_vectors = vectors[i:batch_end]
            
            # Normalize if using cosine similarity
            if self.index_type == "cosine":
                batch_vectors = batch_vectors / np.linalg.norm(batch_vectors, axis=1)[:, np.newaxis]
            
            # Add to FAISS index
            self.index.add(batch_vectors)
            
            # Process metadata
            batch_ids = list(range(self.next_id, self.next_id + len(batch_vectors)))
            batch_metadata = metadata[i:batch_end] if metadata else [{}] * len(batch_vectors)
                
            # Store metadata
            for vid, meta in zip(batch_ids, batch_metadata):
                # Convert VectorMetadata to dict if needed
                meta_dict = meta.to_dict() if isinstance(meta, VectorMetadata) else (meta or {})
                # Merge with defaults
                meta_dict = {**self.default_metadata.to_dict(), **meta_dict}
                meta_dict["vector_id"] = vid
                self.metadata[vid] = VectorMetadata.from_dict(meta_dict)
                
            assigned_ids.extend(batch_ids)
            self.next_id += len(batch_vectors)
            
        return assigned_ids

    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None
    ) -> List[VectorMetadata]:
        """Search for similar vectors."""
        # Ensure vector is 2D
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
            
        # Normalize if using cosine similarity
        if self.index_type == "cosine":
            vector = vector / np.linalg.norm(vector)
            
        # Search in FAISS index
        distances, indices = self.index.search(vector, k)
        
        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not enough results
                continue
                
            # Get metadata
            meta = self.metadata[int(idx)]
            
            # Apply filter if provided
            if filter:
                meta_dict = meta.to_dict()
                if not all(meta_dict.get(k) == v for k, v in filter.items()):
                    continue
                    
            # Convert cosine similarity to distance if needed
            if self.index_type == "cosine":
                # Convert from [-1, 1] similarity to [0, 2] distance
                dist = 1.0 - float(dist)
            else:
                dist = float(dist)
                
            # Add distance to metadata
            meta.distance = dist
            results.append(meta)
            
        # Sort by distance
        results.sort(key=lambda x: x.distance)
        return results

    def get_stats(self) -> Dict[str, Union[int, str]]:
        """Get statistics about the store."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "distance": "cosine" if self.index_type == "cosine" else "l2",
            "memory_usage_mb": self._get_memory_usage()["rss"] / 1024 / 1024
        }

    def _get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
        
    def save(self, save_dir: Optional[str] = None) -> None:
        """Save the vector store to disk."""
        save_dir = save_dir or self.save_dir
        if not save_dir:
            raise ValueError("No save directory specified")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        
        # Save metadata and configuration
        config = {
            "metadata": {str(k): v.to_dict() for k, v in self.metadata.items()},
            "next_id": self.next_id,
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, load_dir: str) -> "FaissStore":
        """Load a vector store from disk."""
        # Load configuration
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found in {load_dir}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Create instance
        instance = cls(
            dimension=config["dimension"],
            save_dir=load_dir
        )
        
        # Load FAISS index
        index_path = os.path.join(load_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index file found in {load_dir}")
            
        instance.index = faiss.read_index(index_path)
        
        # Restore metadata
        instance.metadata = {int(k): VectorMetadata.from_dict(v) for k, v in config["metadata"].items()}
        instance.next_id = config["next_id"]
        
        return instance 