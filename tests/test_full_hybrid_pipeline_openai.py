# tests/test_full_hybrid_pipeline_openai.py

import numpy as np
import pytest
import os
from dotenv import load_dotenv
from rag.embeddings.providers import TextEmbedder
from rag.storage.faiss_store import FaissStore  # Correct class name
from rag.processing.text_chunker import TextChunker
from rag.core.models.vector_store_models import VectorMetadata
from rag.core.config import RAGConfig, EmbeddingConfig
from rag.storage.hybrid_store import HybridStore

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def embedder():
    # Use OpenAI embeddings
    # Note: this requires an OpenAI API key set as OPENAI_API_KEY in environment variables
    # Also requires ENABLE_OPENAI_API=true in environment
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    if os.getenv("ENABLE_OPENAI_API", "false").lower() != "true":
        pytest.skip("ENABLE_OPENAI_API is not set to true")
        
    print(f"OpenAI API Key found: {os.getenv('OPENAI_API_KEY')[:5]}...")
    print(f"ENABLE_OPENAI_API: {os.getenv('ENABLE_OPENAI_API')}")
        
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",  # OpenAI's embedding model
            dimension=1536  # text-embedding-3-small has 1536 dimensions
        )
    )
    return TextEmbedder(config=config)

@pytest.fixture
def vector_store():
    config = RAGConfig()
    config.vector_store.faiss.index_type = "cosine"
    config.vector_store.qdrant.collection_name = "test_openai_hybrid"
    config.vector_store.qdrant.url = "http://localhost:6333"
    return HybridStore(
        dimension=1536,
        config=config
    )

def test_full_pipeline(embedder, vector_store):
    print("\n===== FULL RAG PIPELINE TEST WITH OPENAI EMBEDDER AND FAISS STORE =====\n")
    
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
    print("\nStep 4: Creating metadata with additional fields...")
    metadata = []
    for i, chunk in enumerate(all_chunks):
        # Add extra metadata fields for better filtering
        contains_diabetes = "diabetes" in chunk.lower()
        contains_headache = "headache" in chunk.lower()
        contains_allergy = "allergic" in chunk.lower() or "allergy" in chunk.lower()
        
        # Extract keywords (simple extraction for demo)
        keywords = []
        for term in ["diabetes", "hypertension", "headache", "allergy", "heart disease", "medication"]:
            if term in chunk.lower():
                keywords.append(term)
        
        # Determine document type (simple classification for demo)
        if "prescribed" in chunk.lower() or "medication" in chunk.lower() or "take" in chunk.lower():
            doc_type = "medication"
        elif "allergic" in chunk.lower() or "reaction" in chunk.lower():
            doc_type = "allergy"
        elif "history" in chunk.lower():
            doc_type = "medical_history"
        elif "headache" in chunk.lower() or "pain" in chunk.lower():
            doc_type = "symptom"
        else:
            doc_type = "general"
        
        meta = VectorMetadata(
            text=chunk,
            source="test",
            embedding_model=embedder.model_name
        )
        
        # Add our custom fields directly to the metadata dictionary
        meta.metadata["keywords"] = keywords
        meta.metadata["doc_type"] = doc_type
        meta.metadata["contains_diabetes"] = contains_diabetes
        meta.metadata["contains_headache"] = contains_headache
        meta.metadata["contains_allergy"] = contains_allergy
        
        metadata.append(meta)
    
    print(f"- Created {len(metadata)} metadata objects with enhanced fields")
    # Print an example of the metadata object
    print("- Example metadata structure:")
    print(f"  Text: {metadata[0].text[:50]}...")
    print(f"  Doc Type: {metadata[0].metadata.get('doc_type')}")
    print(f"  Keywords: {metadata[0].metadata.get('keywords')}")
    print(f"  Contains Diabetes: {metadata[0].metadata.get('contains_diabetes')}")
    
    # Add embeddings to the Vector Store
    print("\nStep 5: Storing embeddings in FAISS Store...")
    
    ids = vector_store.add_vectors(embeddings_np, metadata)
    print(f"- Stored {len(ids)} vectors with IDs: {ids[:5]}...")
    stats = vector_store.get_stats()
    print(f"- Vector store stats: {stats}")

    # TEST 1: Basic similarity search
    print("\nTEST 1: Basic similarity search with first text as query")
    query_text = "What medications should I take for diabetes?"
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = vector_store.search_vector(query_vector, k=5)
    
    # Assertions
    assert results is not None
    assert len(results) > 0
    
    # Print results for manual inspection
    print(f"\nQuery: '{query_text}'")
    print("\nSimilarity search results (ordered by relevance):")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
        print(f"  Doc Type: {result.metadata.get('doc_type', 'N/A')}")
        print(f"  Keywords: {result.metadata.get('keywords', [])}")
        print(f"  Contains Diabetes: {result.metadata.get('contains_diabetes', False)}")
        
    # TEST 2: Different query
    print("\nTEST 2: Using a different query (headache search)")
    query_text = "I'm experiencing severe headaches with light sensitivity"
    query_embedding = embedder.embed_text(query_text)
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    results = vector_store.search_vector(query_vector, k=3)
    
    print(f"\nQuery: '{query_text}'")
    print("\nHeadache search results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
        print(f"  Distance: {getattr(result, 'distance', 'N/A'):.4f}" if hasattr(result, 'distance') else f"  Distance: N/A")
        print(f"  Doc Type: {result.metadata.get('doc_type', 'N/A')}")
        print(f"  Keywords: {result.metadata.get('keywords', [])}")
        print(f"  Contains Headache: {result.metadata.get('contains_headache', False)}")
    
            
    print("\n===== TEST COMPLETED SUCCESSFULLY =====") 