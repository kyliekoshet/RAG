"""
Text embedding module for RAG system.

This module handles the conversion of text chunks into vector embeddings
using either OpenAI's API or local HuggingFace models.
"""

from typing import List, Literal, Optional, Union
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(
        self,
        provider: Optional[Literal["openai", "huggingface"]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu"  # Use "cuda" if you have a GPU
    ):
        """
        Initialize the embedder with either OpenAI or HuggingFace.
        
        Args:
            provider: Which embedding provider to use ("openai" or "huggingface").
                     If None, uses DEFAULT_EMBEDDING_PROVIDER from .env
            model: Model name to use. If None, uses defaults:
                  - OpenAI: "text-embedding-ada-002"
                  - HuggingFace: "all-MiniLM-L6-v2"
            api_key: OpenAI API key (only needed for OpenAI)
            device: Device to run HuggingFace models on ("cpu" or "cuda")
            
        Raises:
            ValueError: If using OpenAI without API key or when API is disabled
        """
        # Load environment settings
        load_dotenv()
        self.api_enabled = os.getenv("ENABLE_OPENAI_API", "false").lower() == "true"
        
        # Determine provider
        if provider is None:
            provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface")
        
        self.provider = provider
        
        if provider == "openai":
            # Check if API is enabled
            if not self.api_enabled:
                raise ValueError("OpenAI API is disabled in .env file. Set ENABLE_OPENAI_API=true to enable.")
            
            # Validate API key first
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("API key required for OpenAI (either as parameter or in .env)")
            
            # Get API key from args or environment
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model_name = model or "text-embedding-ada-002"
            self.client = OpenAI(api_key=self.api_key)
        else:  # huggingface
            self.model_name = model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=device)
        
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text into an embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floating point numbers representing the embedding
        
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            else:  # huggingface
                embedding = self.model.encode(text)
                return embedding.tolist()  # Convert numpy array to list
                
        except Exception as e:
            raise Exception(f"Error getting embedding: {str(e)}")
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embedding vectors in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty or not a list
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")
        
        if not all(isinstance(text, str) and text for text in texts):
            raise ValueError("All elements must be non-empty strings")
        
        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float"
                )
                return [data.embedding for data in response.data]
            else:  # huggingface
                embeddings = self.model.encode(texts)
                return embeddings.tolist()  # Convert numpy arrays to lists
                
        except Exception as e:
            raise Exception(f"Error embedding texts: {str(e)}")

if __name__ == "__main__":
    # Load API key from .env file for OpenAI tests
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    api_enabled = os.getenv("ENABLE_OPENAI_API", "false").lower() == "true"
    
    # Test text
    test_text = "This is a test sentence to check if our embeddings are working."
    test_texts = [
        "First test sentence for batch processing.",
        "Second test sentence with different content.",
        "Third test sentence to verify batch embeddings."
    ]
    
    # Test HuggingFace (free)
    print("\n=== Testing HuggingFace Embeddings (Free) ===")
    hf_embedder = TextEmbedder(provider="huggingface")
    
    print("\nTesting single text embedding:")
    embedding = hf_embedder.embed_text(test_text)
    print(f"Model: {hf_embedder.model_name}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    print("\nTesting batch embedding:")
    embeddings = hf_embedder.embed_texts(test_texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    
    # Test OpenAI if enabled and API key is available
    if api_enabled:
        if api_key:
            print("\n=== Testing OpenAI Embeddings (Paid) ===")
            try:
                openai_embedder = TextEmbedder(provider="openai", api_key=api_key)
                
                print("\nTesting single text embedding:")
                embedding = openai_embedder.embed_text(test_text)
                print(f"Model: {openai_embedder.model_name}")
                print(f"Embedding dimension: {len(embedding)}")
                print(f"First 5 values: {embedding[:5]}")
                
                print("\nTesting batch embedding:")
                embeddings = openai_embedder.embed_texts(test_texts)
                print(f"Number of embeddings: {len(embeddings)}")
                print(f"Embedding dimensions: {len(embeddings[0])}")
            except Exception as e:
                print(f"\nError testing OpenAI embeddings: {str(e)}")
        else:
            print("\nSkipping OpenAI tests (no API key found in .env file)")
    else:
        print("\nSkipping OpenAI tests (API disabled in .env file)")
        
