"""
Text chunker implementation with multiple chunking strategies.
Supports fixed-size, recursive, NLTK-based, and markdown-aware chunking.
"""

from typing import List
import logging
import nltk
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger('item-processor-logger')


class TextChunker:
    """
    Text chunker with support for multiple chunking strategies.
    """
    
    def __init__(self, 
                 chunk_size: int = 300, 
                 chunk_overlap: int = 20,
                 strategy: str = 'recursive',
                 use_markdown_splitting: bool = False):
        """
        Initialize text chunker with strategy.
        
        Args:
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between consecutive chunks
            strategy (str): Chunking strategy ('fixed-size', 'recursive', 'nltk-sentence', 'nltk-paragraphs', '1-chunk')
            use_markdown_splitting (bool): Use markdown-aware separators for recursive splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.use_markdown_splitting = use_markdown_splitting
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks based on the configured strategy.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        logger.info(
            f"Chunking text | strategy={self.strategy} chunk_size={self.chunk_size} "
            f"chunk_overlap={self.chunk_overlap} use_markdown={self.use_markdown_splitting} "
            f"text_length={len(text)}"
        )
        
        if self.strategy == 'fixed-size':
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == 'recursive':
            chunks = self._chunk_recursive(text)
        elif self.strategy == 'nltk-sentence':
            chunks = self._chunk_nltk_sentence(text)
        elif self.strategy == 'nltk-paragraphs':
            chunks = self._chunk_nltk_paragraphs(text)
        elif self.strategy == '1-chunk':
            chunks = [text]
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}, using recursive")
            chunks = self._chunk_recursive(text)
        
        logger.info(f"Chunking complete | chunks_created={len(chunks)}")
        return chunks
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Fixed-size chunking with character-based splitting."""
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.create_documents([text])
        return [chunk.page_content for chunk in chunks]
    
    def _chunk_recursive(self, text: str) -> List[str]:
        """Recursive chunking that respects semantic boundaries."""
        if self.use_markdown_splitting:
            # Markdown-aware separators (in order of priority)
            separators = [
                "\n## ",      # H2 headers
                "\n### ",     # H3 headers
                "\n#### ",    # H4 headers
                "\n---\n",    # Horizontal rules
                "\n\n",       # Paragraphs
                "\n",         # Lines
                " ",          # Words
                ""            # Characters
            ]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=separators
            )
        else:
            # Standard recursive splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        
        chunks = text_splitter.create_documents([text])
        return [chunk.page_content for chunk in chunks]
    
    def _chunk_nltk_sentence(self, text: str) -> List[str]:
        """Chunk by sentence boundaries using NLTK."""
        return nltk.sent_tokenize(text)
    
    def _chunk_nltk_paragraphs(self, text: str) -> List[str]:
        """Chunk by paragraph boundaries using NLTK."""
        return nltk.tokenize.blankline_tokenize(text)

