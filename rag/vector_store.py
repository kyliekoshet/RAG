"""
Vector store module for RAG system.

This module handles storing and retrieving embeddings using FAISS.
Storage Details:
    - Vectors are stored in FAISS index (in-memory by default)
    - Metadata is stored in a Python dictionary (in-memory)
    - Each vector gets a unique ID that links it to its metadata
    - Can be persisted to disk (save/load functionality)
    - Supports efficient batch operations
    - Optional structured metadata schema
"""

from typing import List, Dict, Optional, Union, Set
import numpy as np
import faiss
import os
import json
import psutil
from datetime import datetime
from dataclasses import dataclass, asdict, field
from .config import RAGConfig, default_config, EmbeddingConfig, VectorStoreConfig
from .embeddings import TextEmbedder

@dataclass
class VectorMetadata:
    """Metadata for a single vector."""
    text: Optional[str] = None  # Original text used to generate vector
    source: Optional[str] = None  # Where the text came from
    chunk_id: Optional[int] = None  # ID of chunk in document
    doc_id: Optional[str] = None  # Document-level identifier
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())  # When vector was created
    page_number: Optional[int] = None  # If from PDF/book
    tags: Optional[List[str]] = None  # Keywords or topics
    user_id: Optional[str] = None  # If vectors are user-specific
    language: Optional[str] = None  # Language of text
    embedding_model: Optional[str] = None  # Model used to create vector
    vector_id: Optional[int] = None  # ID of the vector in the store
    distance: Optional[float] = None  # Distance when used in search results
    custom_metadata: Optional[Dict] = field(default_factory=dict)  # Any additional metadata

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict) -> 'VectorMetadata':
        """Create from dictionary, handling unknown fields."""
        known_fields = cls.__dataclass_fields__.keys()
        standard_fields = {k: v for k, v in data.items() if k in known_fields}
        custom_fields = {k: v for k, v in data.items() if k not in known_fields}
        
        if custom_fields:
            standard_fields['custom_metadata'] = custom_fields
            
        return cls(**standard_fields)

@dataclass
class VectorBatch:
    """A batch of vectors with their metadata."""
    vectors: np.ndarray
    metadata: List[VectorMetadata]
    
    def __post_init__(self):
        """Validate the batch."""
        if len(self.vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array")
        if len(self.metadata) != len(self.vectors):
            raise ValueError("Number of metadata entries must match number of vectors")

class VectorStore:
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        config: Optional[RAGConfig] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors to store (must match your embeddings)
            index_type: Type of FAISS index to use ('l2' or 'cosine')
                       - 'l2': Euclidean distance (straight-line distance between vectors)
                              Example: distance([1,1], [4,5]) = sqrt((1-4)² + (1-5)²) = 5
                              Smaller distance = more similar
                              
                       - 'cosine': Angular similarity (angle between vectors)
                              Example: similarity([1,1], [2,2]) = 1.0 (parallel vectors)
                              Range: -1 (opposite) to 1 (identical)
                              Better for text embeddings as it ignores magnitude
            save_dir: Directory to save/load the index and metadata
            default_metadata: Default metadata to apply to all vectors
            embedder: TextEmbedder instance for automatic model tracking
            config: RAGConfig instance. If None, uses default_config
            dimension: Override dimension from config/embedder
        """
        self.config = config or default_config
        self.embedder = embedder
        
        # Determine dimension
        if dimension is not None:
            self.dimension = dimension
        elif embedder is not None:
            self.dimension = embedder.dimension
        else:
            self.dimension = self.config.embedding.dimension
            
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
            
        # Get index type from config
        self.index_type = self.config.vector_store.index_type
        self.save_dir = self.config.vector_store.save_dir
        
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
        default_meta = {}
        if embedder:
            default_meta["embedding_model"] = embedder.model_name
        self.default_metadata = VectorMetadata.from_dict(default_meta)

    def _process_metadata(self, meta: Optional[Dict]) -> VectorMetadata:
        """Process metadata dict into schema, applying defaults."""
        if meta is None:
            return self.default_metadata
            
        # Start with defaults, then update with provided metadata
        processed = VectorMetadata.from_dict({
            **self.default_metadata.to_dict(),
            **meta
        })
        
        # Set timestamp if not provided
        if not processed.timestamp:
            processed.timestamp = datetime.now().isoformat()
            
        return processed

    def _create_batch(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        embedding_model: Optional[str] = None
    ) -> VectorBatch:
        """Create a batch of vectors with metadata."""
        if metadata is None:
            metadata = [{}] * len(vectors)
            
        # Process metadata with embedding model
        processed_metadata = []
        for meta in metadata:
            meta_copy = meta.copy() if meta else {}
            if embedding_model:
                meta_copy['embedding_model'] = embedding_model
            processed_metadata.append(self._process_metadata(meta_copy))
            
        return VectorBatch(vectors=vectors, metadata=processed_metadata)

    def add_vector(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a single vector to the store."""
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
            
        ids = self.add_vectors(
            vectors=vector,
            metadata=[metadata] if metadata is not None else None
        )
        return ids[0]

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
        batch_size = self.config.vector_store.batch_size
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
            if metadata is not None:
                batch_metadata = metadata[i:batch_end]
            else:
                batch_metadata = [{}] * len(batch_vectors)
                
            # Store metadata
            for vid, meta in zip(batch_ids, batch_metadata):
                processed_meta = self._process_metadata(meta)
                processed_meta.vector_id = vid
                self.metadata[vid] = processed_meta
                
            assigned_ids.extend(batch_ids)
            self.next_id += len(batch_vectors)
            
        return assigned_ids

    def search_vector(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[callable] = None,
        metadata_fields: Optional[Set[str]] = None
    ) -> tuple[np.ndarray, List[VectorMetadata]]:
        """
        Search for similar vectors to a single query vector.
        
        Args:
            query_vector: numpy array of shape (dimension,)
            k: number of nearest neighbors to return
            filter_fn: optional function to filter results based on metadata
            metadata_fields: optional set of metadata fields to include in results
            
        Returns:
            Tuple of (distances array, list of metadata dictionaries)
        """
        # Ensure query vector is 2D
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Search via batch method
        distances, all_metadata = self.search_vectors(
            query_vectors=query_vector,
            k=k,
            filter_fn=filter_fn,
            metadata_fields=metadata_fields
        )
        return distances[0], all_metadata[0]

    def search_vectors(
        self,
        query_vectors: np.ndarray,
        k: int = 5,
        batch_size: int = 1000,
        filter_fn: Optional[callable] = None,
        metadata_fields: Optional[Set[str]] = None
    ) -> tuple[np.ndarray, List[List[VectorMetadata]]]:
        """
        Search for similar vectors in batches.
        
        Args:
            query_vectors: numpy array of shape (n, dimension)
            k: number of nearest neighbors to return
            batch_size: batch size for search
            filter_fn: optional function to filter results based on metadata
            metadata_fields: optional set of metadata fields to include in results
            
        Returns:
            Tuple of (distances array, list of metadata dictionaries per query)
        """
        if len(query_vectors.shape) != 2 or query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Query vectors must have shape (n, {self.dimension})")
            
        # Ensure query vectors are contiguous and float32
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
            
        # Normalize query vectors if using cosine similarity
        if self.index_type == "cosine":
            norms = np.linalg.norm(query_vectors, axis=1).reshape(-1, 1)
            norms[norms == 0] = 1  # Prevent division by zero
            query_vectors = query_vectors / norms
            
        n_queries = len(query_vectors)
        all_distances = []
        all_metadata = []
        
        # Process in batches
        for i in range(0, n_queries, batch_size):
            batch_end = min(i + batch_size, n_queries)
            batch_queries = query_vectors[i:batch_end]
            
            try:
                # Search in FAISS index
                batch_distances, batch_indices = self.index.search(batch_queries, k)
                
                # Process results for each query
                for distances, indices in zip(batch_distances, batch_indices):
                    query_results = []
                    query_distances = []
                    
                    # Filter valid indices
                    valid_pairs = [(d, idx) for d, idx in zip(distances, indices) if idx != -1]
                    
                    if not valid_pairs:
                        # No valid results for this query
                        all_metadata.append([])
                        all_distances.append(np.full(k, np.inf))
                        continue
                        
                    # Unzip valid pairs
                    valid_distances, valid_indices = zip(*valid_pairs)
                    
                    for dist, idx in zip(valid_distances, valid_indices):
                        try:
                            meta = self.metadata[int(idx)]
                            
                            # Apply filter if provided
                            if filter_fn and not filter_fn(meta.to_dict()):
                                continue
                                
                            # Create result metadata
                            result_meta = meta
                            if metadata_fields:
                                # Create new metadata with only requested fields
                                result_dict = {k: v for k, v in meta.to_dict().items() 
                                             if k in metadata_fields}
                                result_meta = VectorMetadata.from_dict(result_dict)
                                
                            # Add distance
                            result_meta.distance = float(dist)
                            query_results.append(result_meta)
                            query_distances.append(dist)
                            
                        except KeyError:
                            continue  # Skip if metadata not found
                            
                    # Pad distances if needed
                    if len(query_distances) < k:
                        query_distances.extend([np.inf] * (k - len(query_distances)))
                        
                    all_metadata.append(query_results)
                    all_distances.append(query_distances)
                    
            except Exception as e:
                print(f"Error during batch search: {str(e)}")
                # Return empty results for failed batch
                for _ in range(i, batch_end):
                    all_metadata.append([])
                    all_distances.append(np.full(k, np.inf))
                
        if not all_distances:
            return np.array([]), []
            
        return np.array(all_distances), all_metadata

    def delete_vector(self, vector_id: int) -> None:
        """
        Delete a single vector and its metadata.
        
        Args:
            vector_id: ID of vector to delete
        """
        self.delete_vectors([vector_id])

    def delete_vectors(self, vector_ids: List[int]) -> None:
        """
        Delete multiple vectors and their metadata.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        # Get all vectors and metadata
        keep_mask = np.ones(self.index.ntotal, dtype=bool)
        
        # Mark vectors for deletion
        for i in range(self.index.ntotal):
            if i in vector_ids:
                keep_mask[i] = False
                self.metadata.pop(i, None)
                
        # Get vectors to keep
        vectors_to_keep = []
        metadata_to_keep = []
        
        for old_id in range(self.index.ntotal):
            if keep_mask[old_id]:
                vectors_to_keep.append(self.index.reconstruct(old_id))
                if old_id in self.metadata:
                    metadata_to_keep.append(self.metadata[old_id].to_dict())
                
        # Reset index and metadata
        if self.index_type == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        self.metadata = {}
        self.next_id = 0
        
        # Re-add kept vectors
        if vectors_to_keep:
            vectors_array = np.vstack(vectors_to_keep)
            self.add_vectors(vectors_array, metadata_to_keep)

    def get_metadata_fields(self) -> Set[str]:
        """Get all metadata fields used across vectors."""
        fields = set()
        for meta in self.metadata.values():
            fields.update(meta.to_dict().keys())
        return fields

    def get_metadata_stats(self) -> Dict:
        """Get statistics about metadata usage."""
        stats = {
            "total_vectors": len(self.metadata),
            "fields": {},
            "languages": set(),
            "sources": set(),
            "embedding_models": set()
        }
        
        # Count field usage
        for meta in self.metadata.values():
            meta_dict = meta.to_dict()
            for field, value in meta_dict.items():
                if field not in stats["fields"]:
                    stats["fields"][field] = 0
                stats["fields"][field] += 1
                
                # Collect specific field values
                if field == "language" and value:
                    stats["languages"].add(value)
                elif field == "source" and value:
                    stats["sources"].add(value)
                elif field == "embedding_model" and value:
                    stats["embedding_models"].add(value)
                    
        # Convert sets to lists for JSON serialization
        stats["languages"] = list(stats["languages"])
        stats["sources"] = list(stats["sources"])
        stats["embedding_models"] = list(stats["embedding_models"])
        
        return stats

    def _get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "timestamp": datetime.now().isoformat()
        }
        
    def save(self, save_dir: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            save_dir: Directory to save to (overrides self.save_dir)
            
        Raises:
            ValueError: If no save directory specified
        """
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
            "index_type": self.index_type,
            "memory_usage": self._get_memory_usage(),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, load_dir: str) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            load_dir: Directory to load from
            
        Returns:
            Loaded VectorStore instance
            
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If configuration is invalid
        """
        # Load configuration
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found in {load_dir}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Create instance
        instance = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
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
        
    def get_storage_info(self) -> Dict:
        """
        Get information about what's stored in the vector store.
        
        Returns:
            Dictionary containing:
            - number of vectors stored
            - dimension of vectors
            - index type
            - number of metadata entries
        """
        current_memory = self._get_memory_usage()
        memory_growth = {
            "rss_growth": current_memory["rss"] - self.initial_memory["rss"],
            "vms_growth": current_memory["vms"] - self.initial_memory["vms"],
            "percent_growth": current_memory["percent"] - self.initial_memory["percent"]
        }
        
        return {
            "num_vectors": self.index.ntotal,  # Number of vectors in FAISS index
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_metadata": len(self.metadata),
            "memory_usage_bytes": self.index.ntotal * self.dimension * 4, # Approximate memory usage (4 bytes per float)
            "current_memory": current_memory,
            "memory_growth": memory_growth,
            "save_directory": self.save_dir
        }

if __name__ == "__main__":
    # Test automatic embedding model tracking
    try:
        print("Initializing configuration...")
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
        
        print("Creating embedder...")
        embedder = TextEmbedder(config=config)
        
        print("Creating vector store...")
        store = VectorStore(
            embedder=embedder,
            config=config
        )
        
        # Test with increasing batch sizes
        batch_sizes = [10, 50, 100, 500]  # Added larger batch
        
        for n_vectors in batch_sizes:
            print(f"\n=== Testing with {n_vectors} vectors ===")
            print(f"Creating {n_vectors} test vectors...")
            
            # Create test vectors
            vectors = np.random.random((n_vectors, 384)).astype('float32')
            norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
            norms[norms == 0] = 1
            vectors = vectors / norms
            
            # Create simple metadata
            metadata_list = [{
                "text": f"Vector {i}",
                "source": "test"
            } for i in range(n_vectors)]
            
            # Get initial memory usage
            initial_memory = store._get_memory_usage()
            print(f"Initial memory usage: {initial_memory['rss'] / 1024 / 1024:.2f} MB")
            print(f"Initial vector count: {store.index.ntotal}")
            
            print("\nAdding vectors to store...")
            try:
                # Add vectors in smaller batches
                batch_size = min(100, n_vectors)
                for i in range(0, n_vectors, batch_size):
                    end_idx = min(i + batch_size, n_vectors)
                    batch_vectors = vectors[i:end_idx]
                    batch_metadata = metadata_list[i:end_idx]
                    
                    ids = store.add_vectors(batch_vectors, batch_metadata)
                    print(f"Added batch {i//batch_size + 1}: {len(ids)} vectors")
                    
                # Get memory after adding
                after_add_memory = store._get_memory_usage()
                memory_increase = after_add_memory['rss'] - initial_memory['rss']
                print(f"\nMemory increase after adding: {memory_increase / 1024 / 1024:.2f} MB")
                print(f"Final vector count: {store.index.ntotal}")
                
                print("\nTesting search...")
                # Test with individual queries
                n_queries = min(5, n_vectors)
                print(f"Searching with {n_queries} individual queries...")
                
                for i in range(n_queries):
                    print(f"\nQuery {i+1}:")
                    query = vectors[i].reshape(1, -1)
                    
                    try:
                        distances, results = store.search_vector(
                            query_vector=query,
                            k=3
                        )
                        print(f"Found {len(results)} results")
                        
                        for j, result in enumerate(results[:3]):
                            print(f"Result {j+1}:")
                            print(f"  Distance: {result.distance:.4f}")
                            print(f"  Text: {result.text}")
                            
                    except Exception as e:
                        print(f"Error during search {i+1}: {str(e)}")
                        continue
                        
                # Get final memory
                final_memory = store._get_memory_usage()
                total_increase = final_memory['rss'] - initial_memory['rss']
                print(f"\nFinal memory stats:")
                print(f"  Total increase: {total_increase / 1024 / 1024:.2f} MB")
                print(f"  Current RSS: {final_memory['rss'] / 1024 / 1024:.2f} MB")
                print(f"  Current VMS: {final_memory['vms'] / 1024 / 1024:.2f} MB")
                print(f"  Memory percent: {final_memory['percent']:.2f}%")
                
            except Exception as e:
                print(f"Error with batch size {n_vectors}: {str(e)}")
                break
                
    except Exception as e:
        print(f"Error: {str(e)}") 