"""
Processing stages for document processing.
All stages follow signature: (data: dict, config: dict) -> dict

This consistent signature enables piping and makes stages composable.
"""

from .text_normalization import clean_text, normalize_whitespace, remove_empty_lines
from .chunking import chunk_text, chunk_recursive_with_images, chunk_with_embedded_images, TextChunker
from .ocr import ocr_enhance, describe_images_with_dataloop
from .llm import llm_chunk_semantic, llm_summarize
from utils.upload import upload_to_dataloop

__all__ = [
    # Text Normalization
    'clean_text',
    'normalize_whitespace',
    'remove_empty_lines',
    # Chunking
    'chunk_text',
    'chunk_recursive_with_images',
    'chunk_with_embedded_images',
    'TextChunker',
    # OCR
    'ocr_enhance',
    'describe_images_with_dataloop',
    # LLM
    'llm_chunk_semantic',
    'llm_summarize',
    # Upload
    'upload_to_dataloop',
]
