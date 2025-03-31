"""
Models for vector store metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, get_type_hints

@dataclass
class VectorMetadata:
    """Metadata for a vector in the store."""
    text: str
    source: str
    embedding_model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_id: Optional[int] = None
    
    def __init__(
        self, 
        text: str, 
        source: str, 
        embedding_model: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize metadata with custom fields.
        
        Args:
            text: The text content
            source: Source of the text
            embedding_model: Name of the embedding model
            metadata: Optional dictionary of metadata
            vector_id: Optional vector ID
            **kwargs: Any additional fields to store in metadata
        """
        self.text = text
        self.source = source
        self.embedding_model = embedding_model
        self.vector_id = vector_id
        
        # Initialize metadata dictionary
        self.metadata = metadata or {}
        
        # Add any additional keyword arguments to metadata
        if kwargs:
            self.metadata.update(kwargs)
        
    def __getattr__(self, name: str) -> Any:
        """Allow accessing metadata attributes directly."""
        if "metadata" in self.__dict__ and name in self.metadata:
            return self.metadata[name]
        raise AttributeError(f"'VectorMetadata' object has no attribute '{name}'")
        
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting metadata attributes directly."""
        # Don't intercept direct assignment to core attributes
        core_attributes = set(get_type_hints(VectorMetadata).keys())
        if name in core_attributes:
            super().__setattr__(name, value)
        else:
            # For non-core attributes, store in metadata
            if "metadata" not in self.__dict__:
                self.metadata = {}
            self.metadata[name] = value

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
        
        # Create the core object
        instance = cls(
            text=text,
            source=source,
            embedding_model=embedding_model,
            vector_id=vector_id
        )
        
        # Add remaining items as metadata
        if metadata:
            instance.metadata = metadata
            
        return instance 