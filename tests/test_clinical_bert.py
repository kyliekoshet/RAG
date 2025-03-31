import pytest
from rag.embeddings.clinical_bert import ClinicalBERTEmbedder

@pytest.fixture
def embedder():
    return ClinicalBERTEmbedder()

def test_single_text_embedding(embedder):
    text = "Patient has a history of hypertension and diabetes."
    embedding = embedder.embed_text(text)
    
    assert embedding is not None
    assert len(embedding) > 0  # Ensure the embedding is not empty

def test_multiple_texts_embedding(embedder):
    texts = [
        "Patient has a history of hypertension.",
        "Patient is allergic to penicillin.",
        "Patient has been prescribed metformin."
    ]
    embeddings = [embedder.embed_text(text) for text in texts]
    
    assert embeddings is not None
    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert len(embedding) > 0  # Ensure each embedding is not empty