"""
Models for vector store metadata.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class VectorMetadata:
    """Metadata for a vector in the store."""
    text: str
    source: str
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None
    vector_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        base = {
            "text": self.text,
            "source": self.source,
            "embedding_model": self.embedding_model
        }
        if self.vector_id is not None:
            base["vector_id"] = self.vector_id
        if self.metadata:
            base.update(self.metadata)
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorMetadata":
        """Create from dictionary format."""
        metadata = data.copy()
        text = metadata.pop("text")
        source = metadata.pop("source")
        embedding_model = metadata.pop("embedding_model")
        vector_id = metadata.pop("vector_id", None)
        return cls(
            text=text,
            source=source,
            embedding_model=embedding_model,
            metadata=metadata if metadata else None,
            vector_id=vector_id
        ) 