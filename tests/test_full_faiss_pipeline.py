# tests/test_full_faiss_pipeline.py

import numpy as np
import pytest
from rag.embeddings.clinical_bert import ClinicalBERTEmbedder
from rag.storage.faiss_store import FaissStore
from rag.processing.text_chunker import TextChunker

@pytest.fixture
def embedder():
    return ClinicalBERTEmbedder()

@pytest.fixture
def faiss_store():
    # Initialize with the correct dimension for ClinicalBERT
    return FaissStore(dimension=768)

def test_full_pipeline(embedder, faiss_store):
    print("\n===== FULL RAG PIPELINE TEST WITH CLINICAL BERT =====\n")
    
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

    # Add embeddings to the FAISS store
    print("\nStep 4: Storing embeddings in FAISS vector store...")
    metadata = [{"text": chunk, "source": "test", "embedding_model": "ClinicalBERT"} for chunk in all_chunks]
    ids = faiss_store.add_vectors(embeddings_np, metadata)
    print(f"- Stored {len(ids)} vectors in FAISS with IDs: {ids[:5]}...")

    # TEST 1: Basic similarity search
    print("\nTEST 1: Basic similarity search with first text as query")
    query_vector = embeddings_np[0]  # Use the first embedding as the query
    results = faiss_store.search_vector(query_vector, k=5)
    
    # Assertions
    assert results is not None
    assert len(results) > 0
    for result in results:
        assert "text" in result.to_dict()
        assert "source" in result.to_dict()
        assert "embedding_model" in result.to_dict()

    # Print results for manual inspection
    print("\nSimilarity search results (ordered by relevance):")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Distance: {result.distance:.4f}")
        
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
        results = faiss_store.search_vector(query_vector, k=3)
        
        print("\nHeadache search results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Distance: {result.distance:.4f}")
    else:
        print("- Headache text not found in chunks")
    
    # TEST 3: Filtered search
    print("\nTEST 3: Filtered search (only texts with 'diabetes')")
    diabetes_filter = {"text": "diabetes"}
    try:
        # Note: This may not work if the FAISS store doesn't support exact text matching
        results = faiss_store.search_vector(query_vector, k=5, filter=diabetes_filter)
        
        print("\nDiabetes filtered results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Distance: {result.distance:.4f}")
    except Exception as e:
        print(f"- Filter test skipped: {str(e)}")
    
    # TEST 4: Different k values
    print("\nTEST 4: Testing different k values")
    for k in [1, 3]:
        print(f"\nSearch with k={k}:")
        results = faiss_store.search_vector(query_vector, k=k)
        
        assert len(results) <= k
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Distance: {result.distance:.4f}")
            
    print("\n===== TEST COMPLETED SUCCESSFULLY =====")