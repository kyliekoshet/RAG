# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that processes PDFs and uses them for enhanced LLM responses.

## Project Structure

```
RAG/
├── rag/                    # Main package directory
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   └── models/        # Data models
│   ├── processing/        # Text processing
│   │   ├── pdf_parser.py  # PDF parsing
│   │   └── text_chunker.py# Text chunking
│   ├── embeddings/        # Embedding generation
│   │   └── providers.py   # Embedding providers
│   └── storage/           # Vector storage
│       ├── faiss_store.py # FAISS implementation
│       ├── qdrant_store.py# Qdrant implementation
│       └── hybrid_store.py# Hybrid storage
├── tests/                 # Test directory
├── data/                  # Data directory
│   ├── raw/              # Original PDFs
│   ├── processed/        # Processed text
│   └── vectors/          # Vector storage
├── config/               # Configuration
│   ├── default.env       # Template environment
│   └── settings/        # Environment configs
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── examples/        # Usage examples
│   └── guides/          # User guides
├── setup.py             # Package configuration
└── README.md            # This file
```

## Features

- PDF text extraction with structure preservation
- Intelligent text chunking with metadata
- Multiple embedding providers (OpenAI, HuggingFace)
- Flexible vector storage options:
  - FAISS for fast in-memory search
  - Qdrant for persistent storage
  - Hybrid approach combining both
- Configurable through environment variables
- Comprehensive test coverage

## Technology Stack

- **PDF Processing**: `unstructured[pdf]`
- **Text Processing**: `langchain`
- **Embeddings**: 
  - HuggingFace `sentence-transformers`
  - OpenAI API
- **Vector Storage**:
  - FAISS for in-memory
  - Qdrant for persistence
- **Testing**: `pytest`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kyliekoshet/RAG.git
cd RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install in development mode:
```bash
pip install -e .
```

4. Set up configuration:
```bash
cp config/default.env .env
# Edit .env with your settings
```

## Usage Example

```python
from rag import PDFParser, TextChunker, TextEmbedder, HybridStore

# Initialize components
parser = PDFParser("data/raw/document.pdf")
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
embedder = TextEmbedder()  # Uses HuggingFace by default
store = HybridStore(dimension=384)  # Matches embedder dimension

# Process document
text = parser.extract_text()
chunks = chunker.chunk_text_with_metadata(text)

# Generate embeddings and store
vectors = embedder.embed_texts([chunk["text"] for chunk in chunks])
store.add_vectors(vectors, [chunk["metadata"] for chunk in chunks])

# Search similar content
query = "What is the main topic?"
query_vector = embedder.embed_text(query)
results = store.search_vector(query_vector, k=3)
```

## Development

Run tests:
```bash
pytest tests/ -v
```

## Contributors

Kylie Koshet 