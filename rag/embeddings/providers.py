"""
Text embedding module for RAG system.

This module handles the conversion of text chunks into vector embeddings
using either OpenAI's API or local HuggingFace models.
"""

from typing import List, Optional
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from ..core.config import RAGConfig, default_config

# Check if OpenAI API is enabled
api_enabled = os.getenv("ENABLE_OPENAI_API", "false").lower() == "true"

class TextEmbedder:
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the embedder with configuration.
        
        Args:
            config: RAGConfig instance. If None, uses default_config
            api_key: OpenAI API key (only needed for OpenAI)
            provider: Direct provider specification ("openai" or "huggingface")
            model: Direct model name specification
            
        Raises:
            ValueError: If using OpenAI without API key
        """
        # Use config if provided, otherwise create a fresh one
        if config:
            self.config = config
        else:
            self.config = RAGConfig()  # Create fresh config that will load current env vars
            if provider:
                self.config.embedding.provider = provider
            if model:
                self.config.embedding.model_name = model
        
        self.provider = self.config.embedding.provider
        self.model_name = self.config.embedding.model_name
        
        if self.provider == "openai":
            # Validate API key
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("API key required for OpenAI")
            elif not api_enabled:
                raise ValueError("OpenAI API is disabled")
            
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=self.api_key)
        else:  # huggingface
            self.model = SentenceTransformer(
                self.model_name,
                device=self.config.embedding.device
            )
            
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.config.embedding.dimension

    def embed_text(self, text: str) -> List[float]:
        """Convert a single text into an embedding vector."""
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
                return embedding.tolist()
                
        except Exception as e:
            raise Exception(f"Error getting embedding: {str(e)}")
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts into embedding vectors in a batch."""
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
                return embeddings.tolist()
                
        except Exception as e:
            raise Exception(f"Error embedding texts: {str(e)}")

if __name__ == "__main__":
    # Test with default configuration
    embedder = TextEmbedder()
    
    # Test text
    test_text = "This is a test sentence."
    test_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    print(f"\nUsing {embedder.provider} with model {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")
    
    # Test single embedding
    embedding = embedder.embed_text(test_text)
    print(f"\nSingle embedding shape: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    embeddings = embedder.embed_texts(test_texts)
    print(f"\nBatch embedding count: {len(embeddings)}")
    print(f"Each embedding shape: {len(embeddings[0])}")
        
