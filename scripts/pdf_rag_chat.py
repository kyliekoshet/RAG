#!/usr/bin/env python3
"""
pdf_rag_chat.py

A script to:
- Load a PDF file
- Chunk and embed its text
- Store embeddings in a hybrid vector store (FAISS + Qdrant)
- Allow user to ask questions and retrieve relevant text from the PDF

Usage:
    python pdf_rag_chat.py path/to/file.pdf

Requirements:
    pip install PyPDF2
    (All RAG dependencies from setup.py)
    Qdrant server running (docker run -p 6333:6333 qdrant/qdrant)
    .env file with OpenAI/HuggingFace config
"""
import os
import sys
import numpy as np
from dotenv import load_dotenv
import PyPDF2
from rag.embeddings.providers import TextEmbedder
from rag.processing.text_chunker import TextChunker
from rag.core.models.vector_store_models import VectorMetadata
from rag.core.config import RAGConfig, EmbeddingConfig
from rag.storage.hybrid_store import HybridStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time

# Load environment variables
load_dotenv()

# --- 1. Get PDF path ---
if len(sys.argv) < 2:
    print("Usage: python pdf_rag_chat.py path/to/file.pdf")
    sys.exit(1)
pdf_path = sys.argv[1]
if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

# --- 2. Extract text from PDF ---
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

print(f"Loading PDF: {pdf_path}")
pdf_text = extract_pdf_text(pdf_path)
print(f"Extracted {len(pdf_text)} characters from PDF.")

# --- 3. Chunk the text ---
chunker = TextChunker()
chunks = chunker.chunk_text(pdf_text)
print(f"Chunked PDF into {len(chunks)} chunks.")

# --- 4. Set up embedder ---
provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai").lower()
if provider == "openai":
    model_name = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    dimension = 1536
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)
else:
    model_name = os.getenv("HF_MODEL", "sentence-transformers/all-mpnet-base-v2")
    dimension = 768

config = RAGConfig(
    embedding=EmbeddingConfig(
        provider=provider,
        model_name=model_name,
        dimension=dimension
    )
)
embedder = TextEmbedder(config=config)
print(f"Using embedder: {provider} ({model_name})")

# --- 5. Embed the chunks ---
print("Embedding chunks...")
embeddings = embedder.embed_texts(chunks)
embeddings_np = np.array(embeddings, dtype=np.float32)
print(f"Generated embeddings with shape: {embeddings_np.shape}")

# --- 6. Prepare metadata ---
metadata = []
for i, chunk in enumerate(chunks):
    meta = VectorMetadata(
        text=chunk,
        source=os.path.basename(pdf_path),
        embedding_model=model_name
    )
    meta.metadata["chunk_id"] = i
    metadata.append(meta)

# --- 7. Set up HybridStore ---
collection_name = f"pdf_{os.path.basename(pdf_path).replace('.', '_')}_{model_name.replace('/', '_')}_{dimension}"
store_config = RAGConfig()
store_config.vector_store.faiss.index_type = "cosine"
store_config.vector_store.qdrant.collection_name = collection_name
# We'll use host/port instead of URL to ensure consistent connection method
qdrant_host = "localhost"
qdrant_port = 6333
# Explicitly set these in the config
store_config.vector_store.qdrant.host = qdrant_host
store_config.vector_store.qdrant.port = qdrant_port

# Connect to Qdrant and ensure collection with correct dimension
print(f"Setting up Qdrant collection {collection_name} with dimension {dimension}...")
client = QdrantClient(host=qdrant_host, port=qdrant_port)

# First check for any collections with the same name
if client.collection_exists(collection_name):
    print(f"Found existing collection {collection_name}, checking dimension...")
    try:
        collection_info = client.get_collection(collection_name)
        existing_dim = collection_info.config.params.vectors.size
        print(f"Existing collection has dimension {existing_dim}")
        
        if existing_dim != dimension:
            print(f"Dimension mismatch! Deleting old collection with dimension {existing_dim}")
            client.delete_collection(collection_name=collection_name)
            print(f"Waiting for deletion to complete...")
            
            for _ in range(10):
                if not client.collection_exists(collection_name):
                    print(f"Collection deleted successfully")
                    break
                time.sleep(0.5)
    except Exception as e:
        print(f"Error checking collection: {e}")
        print("Deleting collection to be safe...")
        client.delete_collection(collection_name=collection_name)
        time.sleep(2)  # Wait to ensure deletion completes

# Create fresh collection with correct dimension
if not client.collection_exists(collection_name):
    print(f"Creating new collection with dimension {dimension}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection {collection_name} created successfully")

# Now initialize the HybridStore with explicit host/port
vector_store = HybridStore(
    dimension=dimension,
    config=store_config,
    collection_name=collection_name
)

# --- 8. Store embeddings ---
print("Storing embeddings in HybridStore (FAISS + Qdrant)...")
ids = vector_store.add_vectors(embeddings_np, metadata)
print(f"Stored {len(ids)} vectors.")

# --- 9. Interactive question-answer loop ---
print("\nPDF loaded and indexed! You can now ask questions about its content.")
print("Type 'exit' to quit.\n")
while True:
    user_query = input("Your question: ").strip()
    if user_query.lower() in ("exit", "quit"): break
    query_embedding = embedder.embed_text(user_query)
    query_vector = np.array(query_embedding, dtype=np.float32)
    results = vector_store.search_vector(query_vector, k=3)
    print("\nTop relevant chunks:")
    for i, result in enumerate(results):
        print(f"[{i+1}] {result.text[:300]}{'...' if len(result.text) > 300 else ''}")
        print(f"    (Score: {getattr(result, 'score', 'N/A'):.4f}, Distance: {getattr(result, 'distance', 'N/A'):.4f})")
    print("") 