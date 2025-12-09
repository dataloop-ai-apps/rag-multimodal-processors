"""
Chunking transforms for splitting content into chunks.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import re
from typing import List
import logging
import nltk
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from utils.extracted_data import ExtractedData

logger = logging.getLogger("rag-preprocessor")


class TextChunker:
    """Text chunking with support for multiple strategies."""

    @staticmethod
    def chunk(
        text: str,
        chunk_size: int = 300,
        chunk_overlap: int = 20,
        strategy: str = 'recursive'
    ) -> List[str]:
        """
        Split text into chunks based on strategy.

        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy ('fixed-size', 'recursive', 'nltk-sentence', 'nltk-paragraphs', '1-chunk')

        Returns:
            List of text chunks
        """
        logger.info(f"Chunking | strategy={strategy} size={chunk_size} overlap={chunk_overlap}")

        if strategy == 'fixed-size':
            chunks = TextChunker._chunk_fixed_size(text, chunk_size, chunk_overlap)
        elif strategy == 'recursive':
            chunks = TextChunker._chunk_recursive(text, chunk_size, chunk_overlap)
        elif strategy == 'nltk-sentence':
            chunks = TextChunker._chunk_sentence(text)
        elif strategy == 'nltk-paragraphs':
            chunks = TextChunker._chunk_paragraphs(text)
        elif strategy == '1-chunk':
            chunks = [text] if text else []
        else:
            logger.warning(f"Unknown strategy: {strategy}, using recursive")
            chunks = TextChunker._chunk_recursive(text, chunk_size, chunk_overlap)

        logger.info(f"Chunking complete | chunks={len(chunks)}")
        return chunks

    @staticmethod
    def _chunk_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fixed-size chunking."""
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    @staticmethod
    def _chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive chunking that respects semantic boundaries."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    @staticmethod
    def _chunk_sentence(text: str) -> List[str]:
        """Chunk by sentence boundaries using NLTK."""
        return nltk.sent_tokenize(text)

    @staticmethod
    def _chunk_paragraphs(text: str) -> List[str]:
        """Chunk by paragraph boundaries using NLTK."""
        return nltk.tokenize.blankline_tokenize(text)


# Transform wrappers

def chunk(data: ExtractedData) -> ExtractedData:
    """
    Chunk content using the strategy specified in config.

    Handles strategy selection internally:
    - 'semantic': Uses LLM-based semantic chunking
    - 'recursive' with images: Uses chunk_with_images for page/image association
    - Other strategies: Uses TextChunker (fixed, recursive, sentence, none)
    """
    data.current_stage = "chunking"
    strategy = data.config.chunking_strategy

    if strategy == 'semantic':
        from .llm import llm_chunk_semantic
        return llm_chunk_semantic(data)

    if strategy == 'recursive' and data.has_images():
        return chunk_with_images(data)

    content = data.get_text()
    if not content:
        data.chunks = []
        return data

    data.chunks = TextChunker.chunk(
        text=content,
        chunk_size=data.config.max_chunk_size,
        chunk_overlap=data.config.chunk_overlap,
        strategy=strategy,
    )
    data.metadata['chunking_strategy'] = strategy
    data.metadata['chunk_count'] = len(data.chunks)

    return data


def chunk_with_images(data: ExtractedData) -> ExtractedData:
    """Chunk content and associate images based on page numbers."""
    data.current_stage = "chunking"
    content = data.get_text()

    if not content:
        data.chunks = []
        data.chunk_metadata = []
        return data

    # Extract page positions from content
    page_positions = []
    for match in re.finditer(r'---\s*Page\s+(\d+)\s*---', content, re.IGNORECASE):
        page_positions.append((match.start(), int(match.group(1))))

    chunks = TextChunker.chunk(
        text=content,
        chunk_size=data.config.max_chunk_size,
        chunk_overlap=data.config.chunk_overlap,
        strategy='recursive',
    )

    # Build chunk metadata with page and image associations
    chunk_metadata = []
    for chunk_idx, chunk_text in enumerate(chunks):
        chunk_start = content.find(chunk_text)
        if chunk_start == -1:
            chunk_start = sum(len(c) for c in chunks[:chunk_idx])

        # Find pages for this chunk
        chunk_pages = set()
        for pos, page_num in page_positions:
            if pos <= chunk_start:
                chunk_pages.add(page_num)
            elif pos > chunk_start + len(chunk_text):
                break

        if not chunk_pages and page_positions:
            for pos, page_num in reversed(page_positions):
                if pos < chunk_start:
                    chunk_pages.add(page_num)
                    break

        # Find images for this chunk's pages
        image_indices = []
        for img_idx, img in enumerate(data.images):
            if img.page_number and img.page_number in chunk_pages:
                image_indices.append(img_idx)

        chunk_metadata.append({
            'chunk_index': chunk_idx,
            'page_numbers': sorted(list(chunk_pages)) if chunk_pages else None,
            'image_indices': image_indices,
        })

    data.chunks = chunks
    data.chunk_metadata = chunk_metadata
    data.metadata['chunking_strategy'] = 'recursive_with_images'
    data.metadata['chunk_count'] = len(chunks)

    return data
