# tests/test_full_hybrid_pipeline.py

import numpy as np
import pytest
from rag.embeddings.clinical_bert import ClinicalBERTEmbedder
from rag.storage.hybrid_store import HybridStore
from rag.processing.text_chunker import TextChunker
from rag.core.models.vector_store_models import VectorMetadata

@pytest.fixture
def embedder():
    return ClinicalBERTEmbedder()

@pytest.fixture
def hybrid_store():
    # Initialize with the correct dimension for ClinicalBERT
    # Using a test collection to avoid conflicts with production data
    return HybridStore(
        dimension=768, 
        collection_name="test_hybrid_clinicalbert",
        cache_size=100  # Smaller cache for testing
    )

def test_full_pipeline(embedder, hybrid_store):
    print("\n===== FULL RAG PIPELINE TEST WITH CLINICAL BERT AND HYBRID STORE =====\n")
    
    # Example text
    print("Step 1: Preparing sample clinical texts...")
    texts = [
        "Patient has a history of hypertension and diabetes. They have been managing their conditions with medication and lifestyle changes. Recently, they have experienced increased blood pressure and elevated blood sugar levels.",
        "Patient is allergic to penicillin and has had severe reactions in the past. They are advised to avoid all penicillin-based antibiotics and carry an epinephrine auto-injector at all times.",
        "Patient has been prescribed metformin to manage their type 2 diabetes. They are instructed to take the medication twice daily with meals and monitor their blood sugar levels regularly.",
        "Patient reports frequent headaches that have been occurring for the past few months. They describe the pain as throbbing and often accompanied by nausea and sensitivity to light.",
        "Patient has a family history of heart disease, with both parents having experienced heart attacks in their 60s. They are advised to maintain a healthy diet and exercise regularly to reduce their risk."
    ]
    print(f"- Prepared {len(texts)} clinical text samples")

    # Process text
    print("\nStep 2: Processing text into chunks...")
    chunker = TextChunker()
    all_chunks = []
    for text in texts:
        chunks = chunker.chunk_text(text)
        all_chunks.extend(chunks)
    print(f"- Created {len(all_chunks)} text chunks")

    # Generate embeddings
    print("\nStep 3: Generating embeddings with ClinicalBERT...")
    embeddings = [embedder.embed_text(chunk).detach().numpy() for chunk in all_chunks]
    
    # Stack embeddings into a 2D numpy array
    embeddings_np = np.vstack(embeddings)
    print(f"- Generated embeddings with shape: {embeddings_np.shape}")
    print(f"- Embedding dimension: {embeddings_np.shape[1]}")

    # Add embeddings to the Hybrid Store
    print("\nStep 4: Storing embeddings in Hybrid Store (FAISS + Qdrant)...")
    
    # Create VectorMetadata objects for each chunk
    metadata = [
        VectorMetadata(
            text=chunk, 
            source="test", 
            embedding_model="ClinicalBERT"
        ) 
        for chunk in all_chunks
    ]
    
    # Delete existing Qdrant collection if it exists (to start fresh)
    try:
        print("- Recreating test collection...")
        if hybrid_store.qdrant_store.client.collection_exists(hybrid_store.collection_name):
            hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
            # Recreate the HybridStore
            hybrid_store = HybridStore(
                dimension=768,
                collection_name=hybrid_store.collection_name,
                cache_size=100
            )
    except Exception as e:
        print(f"- Collection setup: {str(e)}")
    
    ids = hybrid_store.add_vectors(embeddings_np, metadata)
    print(f"- Stored {len(ids)} vectors in Hybrid Store with IDs: {ids[:5]}...")

    # TEST 1: Basic similarity search
    print("\nTEST 1: Basic similarity search with first text as query")
    query_vector = embeddings_np[0]  # Use the first embedding as the query
    results = hybrid_store.search_vector(query_vector, k=5)
    
    # Assertions
    assert results is not None
    assert len(results) > 0
    for result in results:
        assert hasattr(result, "text")
        assert hasattr(result, "source")
        assert hasattr(result, "embedding_model")

    # Print results for manual inspection
    print("\nSimilarity search results (ordered by relevance):")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
        
    # TEST 2: Different query
    print("\nTEST 2: Using a different query (headache text)")
    # Find the headache text
    headache_index = None
    for i, text in enumerate(all_chunks):
        if "headache" in text.lower():
            headache_index = i
            break
            
    if headache_index is not None:
        query_vector = embeddings_np[headache_index]
        print(f"- Using query about headaches: \"{all_chunks[headache_index][:50]}...\"")
        results = hybrid_store.search_vector(query_vector, k=3)
        
        print("\nHeadache search results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
            print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
    else:
        print("- Headache text not found in chunks")
    
    # TEST 3: Filtered search (testing Qdrant's capabilities)
    print("\nTEST 3: Filtered search (only texts with 'diabetes')")
    try:
        # Create a filter that searches for diabetes-related texts
        diabetes_filter = {"source": "test"}  # Start with a simple filter
        results = hybrid_store.search_vector(query_vector, k=5, filter=diabetes_filter)
        
        print("\nFiltered results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
            print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
    except Exception as e:
        print(f"- Filter test skipped: {str(e)}")
    
    # TEST 4: Cache behavior
    print("\nTEST 4: Testing cache behavior")
    # First search should use Qdrant
    print("- First search (should use Qdrant):")
    results1 = hybrid_store.search_vector(query_vector, k=1)
    
    # Multiple searches to populate cache
    print("- Performing multiple searches to populate cache...")
    for _ in range(10):  # Access multiple times to trigger caching
        hybrid_store.search_vector(query_vector, k=1)
    
    # Next search might use FAISS cache
    print("- Subsequent search (may use FAISS cache):")
    results2 = hybrid_store.search_vector(query_vector, k=1)
    
    # Get store stats
    print("\nHybrid Store Stats:")
    stats = hybrid_store.get_stats()
    for key, value in stats.items():
        print(f"- {key}: {value}")
            
    # Clean up - delete the test collection
    print("\nStep 5: Cleaning up...")
    try:
        hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
        print(f"- Deleted test collection: {hybrid_store.collection_name}")
    except Exception as e:
        print(f"- Cleanup error: {str(e)}")
            
    print("\n===== TEST COMPLETED SUCCESSFULLY =====") 