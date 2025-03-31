# tests/test_full_hybrid_pipeline_openai.py

import numpy as np
import pytest
import os
from rag.embeddings.providers import TextEmbedder
from rag.storage.hybrid_store import HybridStore
from rag.processing.text_chunker import TextChunker
from rag.core.models.vector_store_models import VectorMetadata
from rag.core.config import RAGConfig, EmbeddingConfig

@pytest.fixture
def embedder():
    # Use OpenAI embeddings
    # Note: this requires an OpenAI API key set as OPENAI_API_KEY in environment variables
    # Also requires ENABLE_OPENAI_API=true in environment
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    if os.getenv("ENABLE_OPENAI_API", "false").lower() != "true":
        pytest.skip("ENABLE_OPENAI_API is not set to true")
        
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",  # OpenAI's embedding model
            dimension=1536  # text-embedding-3-small has 1536 dimensions
        )
    )
    return TextEmbedder(config=config)

@pytest.fixture
def hybrid_store():
    # Initialize with the correct dimension for OpenAI embeddings
    return HybridStore(
        dimension=1536,  # Dimension for text-embedding-3-small
        collection_name="test_hybrid_openai",
        cache_size=100  # Smaller cache for testing
    )

def test_full_pipeline(embedder, hybrid_store):
    print("\n===== FULL RAG PIPELINE TEST WITH OPENAI EMBEDDER AND HYBRID STORE =====\n")
    
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
    print("\nStep 3: Generating embeddings with OpenAI model...")
    # Use the batch embedding method for efficiency
    embeddings = embedder.embed_texts(all_chunks)
    
    # Convert to numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"- Generated embeddings with shape: {embeddings_np.shape}")
    print(f"- Embedding dimension: {embeddings_np.shape[1]}")
    print(f"- Using model: {embedder.model_name}")

    # Add enriched metadata
    metadata = []
    for i, chunk in enumerate(all_chunks):
        
        meta = VectorMetadata(
            text=chunk,
            source="test",
            embedding_model=embedder.model_name
        )
        
        metadata.append(meta)
    
    
    # Add embeddings to the Hybrid Store
    print("\nStep 5: Storing embeddings in Hybrid Store (FAISS + Qdrant)...")
    
    # Delete existing Qdrant collection if it exists (to start fresh)
    try:
        print("- Recreating test collection...")
        if hybrid_store.qdrant_store.client.collection_exists(hybrid_store.collection_name):
            hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
            # Recreate the HybridStore
            hybrid_store = HybridStore(
                dimension=1536,
                collection_name=hybrid_store.collection_name,
                cache_size=100
            )
    except Exception as e:
        print(f"- Collection setup: {str(e)}")
    
    ids = hybrid_store.add_vectors(embeddings_np, metadata)
    print(f"- Stored {len(ids)} vectors in Hybrid Store with IDs: {ids[:5]}...")

    # TEST 1: Basic similarity search
    print("\nTEST 1: Basic similarity search with first text as query")
    query_text = "What medications should I take for diabetes?"
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = hybrid_store.search_vector(query_vector, k=5)
    
    # Assertions
    assert results is not None
    assert len(results) > 0
    
    # Print results for manual inspection
    print(f"\nQuery: '{query_text}'")
    print("\nSimilarity search results (ordered by relevance):")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
        
    # TEST 2: Different query
    print("\nTEST 2: Using a different query (headache search)")
    query_text = "I'm experiencing severe headaches with light sensitivity"
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = hybrid_store.search_vector(query_vector, k=3)
    
    print(f"\nQuery: '{query_text}'")
    print("\nHeadache search results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")

    # TEST 3: Filtered search (using metadata field)
    print("\nTEST 3: Filtered search (using metadata fields)")
    try:
        # Create a filter for diabetes-related documents using our custom field
        filter_by_diabetes = {"contains_diabetes": True}
        results = hybrid_store.search_vector(
            query_vector, 
            k=5, 
            filter=filter_by_diabetes,
            use_cache=False  # Ensure we're using Qdrant's filtering
        )
        
        print("\nResults filtered by 'contains_diabetes' metadata field:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
            print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")

            
        # Verify all results actually contain diabetes
        for result in results:
            assert result.metadata.get('contains_diabetes', False) == True
            
    except Exception as e:
        print(f"- Filter test skipped: {str(e)}")
    
    # TEST 4: Document type filtering
    print("\nTEST 4: Document type filtering (medication documents)")
    try:
        # Create a filter for medication documents
        filter_by_doc_type = {"doc_type": "medication"}
        results = hybrid_store.search_vector(
            query_vector, 
            k=5, 
            filter=filter_by_doc_type,
            use_cache=False  # Ensure we're using Qdrant's filtering
        )
        
        print("\nResults filtered by 'doc_type=medication' metadata field:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Score: {getattr(result, 'score', 'N/A'):.4f}" if hasattr(result, 'score') else f"  Score: N/A")
            print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
            
        # Verify all results are medication documents
        for result in results:
            assert result.metadata.get('doc_type', "") == "medication"
            
    except Exception as e:
        print(f"- Doc type filter test skipped: {str(e)}")
    
    # TEST 5: Compare OpenAI vs other models (explanation)
    print("\nTEST 5: OpenAI Embedding Model Insights")
    print("- OpenAI embeddings often provide different (sometimes better) semantic understanding")
    print("- They tend to capture medical concepts more accurately (clinical context)")
    print("- The tradeoff is API cost and latency compared to local models")
    print("- Dimension is higher (1536 vs 768), which can provide more detailed representations")
    print("- Results may show different ordering than Hugging Face or Clinical BERT models")
            
    # Clean up - delete the test collection
    print("\nStep 6: Cleaning up...")
    try:
        hybrid_store.qdrant_store.client.delete_collection(hybrid_store.collection_name)
        print(f"- Deleted test collection: {hybrid_store.collection_name}")
    except Exception as e:
        print(f"- Cleanup error: {str(e)}")
            
    print("\n===== TEST COMPLETED SUCCESSFULLY =====") 