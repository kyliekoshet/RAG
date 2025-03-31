"""
Qdrant-based vector store for RAG system.

This module provides vector storage and similarity search using Qdrant.
Benefits:
- Optimized for vector similarity search
- Built-in filtering and metadata
- Persistent storage
- Production-ready
- REST API included
"""

from typing import List, Dict, Optional, Union, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText
from ..core.models.vector_store_models import VectorMetadata
from ..core.config import RAGConfig, default_config
from .base_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    """Vector store implementation using Qdrant."""
    
    def __init__(
        self,
        dimension: int,
        config: Optional[RAGConfig] = None,
        collection_name: str = "rag_vectors",
        host: str = "localhost",
        port: int = 6333
    ):
        """Initialize the Qdrant store.
        
        Args:
            dimension: Dimension of vectors
            config: RAG configuration
            collection_name: Name of the collection to use
            host: Qdrant host
            port: Qdrant port
        """
        super().__init__(dimension=dimension, config=config)
        self.collection_name = collection_name
        
        # Initialize client
        self.client = QdrantClient(host=host, port=port)
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
    
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
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata")
            
        # Convert vectors to list of points
        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            points.append(
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload=meta.to_dict()
                )
            )
            
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return [p.id for p in points]
    
    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        content_filter: Optional[str] = None
    ) -> List[VectorMetadata]:
        """Search for similar vectors.
        
        Args:
            vector: Query vector
            k: Number of results to return
            filter: Optional metadata filter
            content_filter: Optional text content to filter by (will search in 'text' field)
            
        Returns:
            List of metadata for similar vectors
        """
        # Convert filter to Qdrant format
        search_filter = None
        if filter or content_filter:
            conditions = []
            
            # Add standard key-value filters
            if filter:
                for key, value in filter.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            # Add text content filter if provided
            if content_filter:
                # Use Qdrant's text match capabilities to filter by text content
                conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=content_filter)
                    )
                )
                
            search_filter = Filter(
                must=conditions
            )
        
        # Search using query_points
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=k,
            query_filter=search_filter
        )
        
        # Convert results to metadata
        metadata_results = []
        for hit in results:
            # Create VectorMetadata object from payload
            metadata = VectorMetadata.from_dict(hit.payload)
            
            # Add score as a property
            metadata.score = hit.score
            
            # Calculate distance (1 - score for cosine similarity)
            metadata.distance = 1.0 - hit.score
            
            metadata_results.append(metadata)
            
        return metadata_results
    
    def get_stats(self) -> Dict[str, Union[int, str]]:
        """Get statistics about the store.
        
        Returns:
            Dictionary with stats
        """
        collection = self.client.get_collection(self.collection_name)
        return {
            "total_vectors": collection.points_count,
            "dimension": collection.config.params.vectors.size,
            "distance": collection.config.params.vectors.distance
        }

if __name__ == "__main__":
    # Test the Qdrant store
    try:
        print("Initializing Qdrant store...")
        store = QdrantStore(
            dimension=384  # Same as your embedding dimension
        )
        
        print("\nCreating test vectors...")
        n_vectors = 10
        vectors = np.random.random((n_vectors, 384)).astype('float32')
        
        # Normalize vectors for cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        
        # Create test metadata
        metadata = [
            VectorMetadata(
                text=f"Test vector {i}",
                source="test",
                embedding_model="test-model"
            )
            for i in range(n_vectors)
        ]
        
        print("Adding vectors...")
        ids = store.add_vectors(vectors, metadata)
        print(f"Added {len(ids)} vectors")
        
        print("\nTesting single vector search...")
        # Use first vector as query
        results = store.search_vector(vectors[0], k=3)
        
        print("\nTop 3 results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text}")
            print(f"  Source: {result.source}")
            print(f"  Model: {result.embedding_model}")
            
        print("\nTesting filtered search...")
        # Search with filter
        filter_results = store.search_vector(
            vectors[0],
            k=3,
            filter={"source": "test"}
        )
        
        print("\nFiltered results:")
        for i, result in enumerate(filter_results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text}")
            print(f"  Source: {result.source}")
            print(f"  Model: {result.embedding_model}")
            
        print("\nGetting stats...")
        stats = store.get_stats()
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Distance metric: {stats['distance']}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 