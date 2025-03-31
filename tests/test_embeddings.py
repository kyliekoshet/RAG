from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
api_enabled = os.getenv("ENABLE_OPENAI_API", "false").lower() == "true"

"""
Tests for the text embedding functionality.
"""

import pytest
from rag.embeddings import TextEmbedder
import os

def test_huggingface_initialization():
    """Test TextEmbedder initialization with HuggingFace (default)."""
    embedder = TextEmbedder()  # Should default to huggingface
    assert embedder.provider == "huggingface"
    assert embedder.model_name == "all-MiniLM-L6-v2"  # default model
    assert hasattr(embedder, "model")

def test_huggingface_with_custom_model():
    """Test TextEmbedder initialization with custom HuggingFace model."""
    custom_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = TextEmbedder(provider="huggingface", model=custom_model)
    assert embedder.model_name == custom_model

def test_openai_initialization_when_disabled():
    """Test TextEmbedder initialization fails when OpenAI is disabled."""
    if api_enabled:
        pytest.skip("OpenAI API is enabled, skipping disabled test")
    
    with pytest.raises(ValueError, match="OpenAI API is disabled"):
        TextEmbedder(provider="openai", api_key="dummy-key")

@pytest.mark.skipif(not api_enabled, reason="OpenAI API is disabled")
def test_openai_initialization_when_enabled():
    """Test TextEmbedder initialization with OpenAI when enabled."""
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")
    
    embedder = TextEmbedder(provider="openai", api_key=api_key)
    assert embedder.provider == "openai"
    assert embedder.model_name == "text-embedding-ada-002"  # default model
    assert hasattr(embedder, "client")

def test_embedder_without_api_key():
    """Test TextEmbedder initialization fails without API key when using OpenAI."""
    if not api_enabled:
        pytest.skip("OpenAI API is disabled")
    
    # Temporarily clear the environment variable
    original_key = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        with pytest.raises(ValueError, match="API key required"):
            TextEmbedder(provider="openai", api_key="")
    finally:
        # Restore the environment variable
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

def test_basic_embedding_huggingface():
    """Test basic text embedding with HuggingFace."""
    embedder = TextEmbedder(provider="huggingface")
    text = "This is a test sentence."
    
    # Test single text
    embedding = embedder.embed_text(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Test batch
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedder.embed_texts(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == len(embeddings[0]) for emb in embeddings)

@pytest.mark.skipif(not api_enabled or not api_key, 
                    reason="OpenAI API is disabled or key not found")
def test_basic_embedding_openai():
    """Test basic text embedding with OpenAI."""
    embedder = TextEmbedder(provider="openai", api_key=api_key)
    text = "This is a test sentence."
    
    # Test single text
    embedding = embedder.embed_text(text)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Test batch
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedder.embed_texts(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == len(embeddings[0]) for emb in embeddings)

def test_invalid_input():
    """Test error handling for invalid inputs."""
    embedder = TextEmbedder()  # Use default HuggingFace
    
    # Test empty text
    with pytest.raises(ValueError):
        embedder.embed_text("")
    
    # Test None
    with pytest.raises(ValueError):
        embedder.embed_text(None)
    
    # Test empty list
    with pytest.raises(ValueError):
        embedder.embed_texts([])
    
    # Test list with empty string
    with pytest.raises(ValueError):
        embedder.embed_texts(["valid text", "", "also valid"])

def test_environment_default_provider():
    """Test that default provider from environment is respected."""
    # Save original value
    original_provider = os.getenv("EMBEDDING_PROVIDER")

    try:
        # Set environment variable
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"

        # Create embedder and check provider
        embedder = TextEmbedder()
        assert embedder.provider == "huggingface"

        # Only test OpenAI if API is enabled
        if api_enabled:
            # Change provider and verify it's respected
            os.environ["EMBEDDING_PROVIDER"] = "openai"
            embedder = TextEmbedder()
            assert embedder.provider == "openai"

    finally:
        # Restore original value
        if original_provider:
            os.environ["EMBEDDING_PROVIDER"] = original_provider
        else:
            del os.environ["EMBEDDING_PROVIDER"] 