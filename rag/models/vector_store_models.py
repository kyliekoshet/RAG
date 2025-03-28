"""
Data models for the vector store module.

This module contains the data structures used by the vector store,
including metadata schemas and batch operations.
"""

from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field

@dataclass
class VectorMetadata:
    """Metadata for a single vector."""
    text: Optional[str] = None  # Original text used to generate vector
    source: Optional[str] = None  # Where the text came from
    chunk_id: Optional[int] = None  # ID of chunk in document
    doc_id: Optional[str] = None  # Document-level identifier
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())  # When vector was created
    page_number: Optional[int] = None  # If from PDF/book
    tags: Optional[List[str]] = None  # Keywords or topics
    user_id: Optional[str] = None  # If vectors are user-specific
    language: Optional[str] = None  # Language of text
    embedding_model: Optional[str] = None  # Model used to create vector
    vector_id: Optional[int] = None  # ID of the vector in the store
    distance: Optional[float] = None  # Distance when used in search results
    custom_metadata: Optional[Dict] = field(default_factory=dict)  # Any additional metadata

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict) -> 'VectorMetadata':
        """Create from dictionary, handling unknown fields."""
        known_fields = cls.__dataclass_fields__.keys()
        standard_fields = {k: v for k, v in data.items() if k in known_fields}
        custom_fields = {k: v for k, v in data.items() if k not in known_fields}
        
        if custom_fields:
            standard_fields['custom_metadata'] = custom_fields
            
        return cls(**standard_fields)

@dataclass
class VectorBatch:
    """A batch of vectors with their metadata."""
    vectors: np.ndarray
    metadata: List[VectorMetadata]
    
    def __post_init__(self):
        """Validate the batch."""
        if len(self.vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array")
        if len(self.metadata) != len(self.vectors):
            raise ValueError("Number of metadata entries must match number of vectors") 