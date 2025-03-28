from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "unstructured[pdf]>=0.11.8",
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "sentence-transformers>=2.2.0",  # For local embeddings
        "faiss-cpu>=1.7.4",  # For vector storage
        "numpy>=1.24.0",  # Required by FAISS
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
) 