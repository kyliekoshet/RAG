from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "unstructured[pdf]>=0.11.8",
        "langchain>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
) 