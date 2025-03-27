from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union

class TextChunker:
    """
    A class for splitting text into chunks of a specified size.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextChunker with specified chunk size and overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Fix separators (remove duplicates and order from largest to smallest)
        self.separator = [
            "\n\n",     # Paragraph breaks (keep this first)
            "\n",       # Line breaks
            ". ",       # End of sentence
            "! ",       # Exclamation
            "? ",       # Question
            "; ",       # Semicolon
            ": ",       # Colon
            ", ",       # Comma
            " ",        # Space (keep this second to last)
            ""          # Character (keep this last)
        ]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separator
        )
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def chunk_text_with_metadata(self, text: str, source_info: dict = None) -> List[dict]:
        """
        Split text into chunks and add metadata.
        
        Args:
            text: The text to split
            source_info: Dictionary containing source information (e.g., file name, page number)
                
        Returns:
            List of dictionaries containing:
                - text: The chunk text
                - metadata: Information about the chunk
        """
        chunks = self.chunk_text(text)
        result = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = {
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk),
                "chunk_position": "start" if i == 1 else "end" if i == len(chunks) else "middle",
            }
            
            # Add any source information if provided
            if source_info:
                metadata.update(source_info)
                
            result.append({
                "text": chunk,
                "metadata": metadata
            })
        
        return result
        
    def debug_chunks(self, text: str) -> None:
        """Debug the chunking process."""
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i} (length: {len(chunk)} characters):")
            print("Start:", chunk[:50], "...")  # First 50 chars
            print("End:", "...", chunk[-50:])   # Last 50 chars
            
            if i < len(chunks):
                # Look for overlap at the end of this chunk and start of next chunk
                end_of_current = chunk[-self.chunk_overlap:]
                start_of_next = chunks[i][:self.chunk_overlap]
                print("\nOverlap check:")
                print("End of current chunk:", end_of_current)
                print("Start of next chunk:", start_of_next)
                common = set(end_of_current.split()).intersection(set(start_of_next.split()))
                if common:
                    print("Common words:", common)
            print("-" * 100)

if __name__ == "__main__":
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit leo venenatis, maecenas egestas scelerisque nisl aptent erat sodales iaculis curabitur, hendrerit tristique porttitor ornare semper imperdiet vel a. Varius venenatis convallis urna platea praesent mus donec nunc, dis magnis volutpat nostra euismod duis felis taciti elementum, posuere aptent vehicula ornare viverra quisque congue. Felis euismod litora curae duis nulla quis venenatis sodales, metus hac nibh senectus congue at suscipit, ornare tellus torquent morbi ad posuere habitasse.\n\nUltrices rhoncus a morbi primis phasellus tristique integer, natoque scelerisque ante per nulla sollicitudin egestas habitasse, cubilia himenaeos sociosqu suspendisse neque libero. Orci accumsan proin etiam lobortis nibh gravida tristique nostra hendrerit, nunc tempor libero habitant sagittis potenti aenean eget donec, vulputate nam velit ultricies neque sem ante senectus. Quisque aenean consequat per integer pretium parturient, lacus libero semper phasellus erat lobortis, ad sagittis in sociis convallis."
    
    # Example source info
    source_info = {
        "source_type": "sample_text",
        "document_id": "sample_001",
        "language": "latin"
    }
    
    chunker = TextChunker(chunk_size=200, chunk_overlap=100)
    chunks_with_metadata = chunker.chunk_text_with_metadata(text, source_info)
    
    # Print chunks with their metadata
    for chunk_info in chunks_with_metadata:
        print("\n" + "="*50)
        print("CHUNK METADATA:")
        for key, value in chunk_info["metadata"].items():
            print(f"{key}: {value}")
        print("\nCHUNK TEXT:")
        print(chunk_info["text"][:100] + "..." if len(chunk_info["text"]) > 100 else chunk_info["text"])
        print("="*50)