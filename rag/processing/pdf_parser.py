"""
PDF Parser module for RAG system.

This module provides functionality to extract clean, structured text from PDF documents
using the unstructured library. It handles PDF processing, text extraction, and basic
cleaning operations.
"""

from pathlib import Path
from typing import List, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Title


class PDFParser:
    """A class to handle PDF parsing and text extraction operations."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDFParser with a path to a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file to be processed
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    def extract_text(self, include_titles: bool = True) -> List[str]:
        """
        Extract text from the PDF file.
        
        Args:
            include_titles (bool): Whether to include titles in the extracted text
                                 Defaults to True.
        
        Returns:
            List[str]: List of extracted text elements from the PDF
        
        Note:
            This method uses unstructured's partition_pdf which automatically handles:
            - Text extraction from different PDF formats
            - Maintaining reading order
            - Basic structure recognition (titles, paragraphs, etc.)
        """
        # Extract elements from PDF using unstructured
        elements = partition_pdf(
            filename=str(self.pdf_path),
            strategy="fast",  # Use fast strategy for better performance
        )
        
        # Filter and clean the extracted text
        extracted_text = []
        for element in elements:
            # Include Title elements if include_titles is True
            if isinstance(element, Title) and include_titles:
                extracted_text.append(str(element))
            # Always include regular Text elements
            elif isinstance(element, Text):
                extracted_text.append(str(element))
        
        return self._clean_text_elements(extracted_text)
    
    def _clean_text_elements(self, text_elements: List[str]) -> List[str]:
        """
        Clean the extracted text elements by removing extra whitespace and empty strings.
        
        Args:
            text_elements (List[str]): List of extracted text elements
            
        Returns:
            List[str]: List of cleaned text elements
        """
        cleaned_elements = []
        for text in text_elements:
            # Strip whitespace and normalize spaces
            cleaned_text = " ".join(text.split())
            if cleaned_text:  # Only include non-empty strings
                cleaned_elements.append(cleaned_text)
        
        return cleaned_elements


def main():
    """
    Example usage of the PDFParser class.
    """
    # Example usage
    try:
        parser = PDFParser("example.pdf")
        extracted_text = parser.extract_text()
        print(f"Successfully extracted {len(extracted_text)} text elements")
        
        # Print first few elements as preview
        for i, text in enumerate(extracted_text[:3], 1):
            print(f"\nElement {i}:")
            print(text[:100] + "..." if len(text) > 100 else text)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 