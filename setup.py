"""
Setup file for RAG package.
"""

from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.2",
        "python-dotenv>=1.0.0",
        "pytest>=8.0.0",
        "openai>=1.12.0",
        "transformers>=4.36.0",
        "torch>=2.1.0"
    ],
    python_requires=">=3.8",
    description="RAG (Retrieval Augmented Generation) system with support for multiple embedding providers",
    author="Kylie Koshet",
    author_email="kylie.koshet@gmail.com",
    url="https://github.com/kyliekoshet/RAG",
) 