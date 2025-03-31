"""
Tests for vector store implementations.
"""

import os
import sys
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.core.config import hybrid_config
from rag.storage import HybridStore, FaissStore, QdrantStore
from rag.core.models.vector_store_models import VectorMetadata

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_vectors():
    """Fixture to create test vectors and metadata."""
    n_vectors = 100  # Smaller set for tests
    
    # Sample texts
    texts = [
        f"This is test document {i} about topic {i % 5}"
        for i in range(n_vectors)
    ]
    
    # Create embeddings
    model = SentenceTransformer(hybrid_config.embedding.model_name)
    vectors = model.encode(texts, convert_to_numpy=True)
    
    # Create metadata
    metadata = [
        VectorMetadata(
            text=text,
            source="test",
            embedding_model=hybrid_config.embedding.model_name,
            metadata={
                "topic": f"topic_{i % 5}",
                "doc_id": f"doc_{i}"
            }
        )
        for i, text in enumerate(texts)
    ]
    
    return vectors, metadata

@pytest.fixture(scope="function")
def hybrid_store(tmp_path):
    """Fixture to create a hybrid store instance."""
    # Use temporary directory for vector store
    config = hybrid_config
    config.vector_store.save_dir = str(tmp_path / "vectors")
    
    store = HybridStore(
        dimension=config.embedding.dimension,
        config=config,
        collection_name=f"test_vectors_{os.getpid()}",  # Unique collection name per process
        cache_size=config.vector_store.faiss.cache_size
    )
    yield store
    
    # Cleanup after test
    try:
        store.qdrant_store._client.delete_collection(store.collection_name)
    except:
        pass

def test_hybrid_store_initialization(hybrid_store):
    """Test that the store initializes correctly."""
    assert hybrid_store.dimension == hybrid_config.embedding.dimension
    assert hybrid_store.cache_size == hybrid_config.vector_store.faiss.cache_size
    assert hybrid_store.collection_name.startswith("test_vectors_")

def test_adding_vectors(hybrid_store, test_vectors):
    """Test adding vectors to the store."""
    vectors, metadata = test_vectors
    
    # Add vectors
    ids = hybrid_store.add_vectors(vectors, metadata)
    
    # Verify
    assert len(ids) == len(vectors)
    assert hybrid_store.qdrant_store.get_stats()["total_vectors"] == len(vectors)
    
    # Check if vectors were added to FAISS cache (should be, as it's under cache_size)
    assert hybrid_store.faiss_store.index.ntotal == len(vectors)

def test_basic_search(hybrid_store, test_vectors):
    """Test basic vector search."""
    vectors, metadata = test_vectors
    
    # Add vectors first
    hybrid_store.add_vectors(vectors, metadata)
    
    # Search
    results = hybrid_store.search_vector(vectors[0], k=3)
    
    # Verify
    assert len(results) == 3
    assert all(hasattr(r, 'distance') for r in results)
    assert all(hasattr(r, 'text') for r in results)
    
    # First result should be the query vector itself
    assert abs(results[0].distance) < 1e-5  # Should be very close to 0

def test_filtered_search(hybrid_store, test_vectors):
    """Test search with metadata filtering."""
    vectors, metadata = test_vectors
    
    # Add vectors
    hybrid_store.add_vectors(vectors, metadata)
    
    # Search with filter
    filter = {"topic": "topic_0"}
    results = hybrid_store.search_vector(vectors[0], k=3, filter=filter)
    
    # Verify
    assert len(results) > 0
    assert all(r.to_dict()["topic"] == "topic_0" for r in results)

def test_cache_behavior(hybrid_store, test_vectors):
    """Test that frequently accessed vectors are cached."""
    vectors, metadata = test_vectors
    
    # Add vectors
    hybrid_store.add_vectors(vectors, metadata)
    
    # Initial cache state
    initial_cache_size = hybrid_store.faiss_store.index.ntotal
    
    # Search same vector multiple times
    test_vector = vectors[0]
    for _ in range(7):  # More than access_threshold (5)
        hybrid_store.search_vector(test_vector, k=1)
    
    # Verify cache was updated
    assert hybrid_store.access_counts[0] >= 5  # Vector 0 was accessed multiple times

def test_stats(hybrid_store, test_vectors):
    """Test statistics reporting."""
    vectors, metadata = test_vectors
    
    # Add vectors
    hybrid_store.add_vectors(vectors, metadata)
    
    # Get stats
    stats = hybrid_store.get_stats()
    
    # Verify
    assert "total_vectors" in stats
    assert "cached_vectors" in stats
    assert "cache_hit_ratio" in stats
    assert stats["total_vectors"] == len(vectors)
    assert 0 <= stats["cache_hit_ratio"] <= 1

@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_different_k_values(hybrid_store, test_vectors, k):
    """Test search with different k values."""
    vectors, metadata = test_vectors
    
    # Add vectors
    hybrid_store.add_vectors(vectors, metadata)
    
    # Search
    results = hybrid_store.search_vector(vectors[0], k=k)
    
    # Verify
    assert len(results) == k
    assert all(hasattr(r, 'distance') for r in results)
    assert all(r.distance >= 0 for r in results)  # Distances should be non-negative
    
    # Results should be sorted by distance
    distances = [r.distance for r in results]
    assert distances == sorted(distances)  # Should be in ascending order 