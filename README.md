# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that processes PDFs and uses them for enhanced LLM responses.

## Project Structure

```
RAG/
├── rag/                      # Main package directory
│   ├── __init__.py          # Makes it a package, exports PDFParser
│   └── pdf_parser.py        # PDF parsing and text extraction
├── tests/                    # Test directory
│   └── test_pdf_parser.py   # Unit tests
├── setup.py                 # Package configuration
└── rag.egg-info/            # Generated metadata (can be ignored)
```

## Features

- PDF text extraction with structure preservation
- Clean text processing
- Title and content separation
- Modular and testable design

## Technology Stack

- **PDF Processing**: `unstructured[pdf]`
- **Testing**: `pytest`
- **Coming Soon**:
  - LangChain for text chunking
  - OpenAI embeddings
  - Qdrant/FAISS vector store
  - FastAPI backend

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

## Running Tests

```bash
pytest tests/ -v
```

## Usage Example

```python
from rag import PDFParser

# Initialize parser with a PDF file
parser = PDFParser("path/to/your/file.pdf")

# Extract text
text = parser.extract_text()

# Extract text without titles
content_only = parser.extract_text(include_titles=False)
```

## Under Development

This project is actively being developed. Next steps include:
- Text chunking implementation
- Vector embeddings integration
- Vector store setup
- API development
- Question-answering functionality


## Contributors

Kylie Koshet 