"""
Processing transforms for document processing.

All transforms follow signature: (data: ExtractedData) -> ExtractedData
"""

from .text_normalization import clean, normalize_whitespace, remove_empty_lines, deep_clean, TextNormalizer
from .chunking import chunk, chunk_with_images, TextChunker
from .ocr import ocr_enhance, describe_images, OCREnhancer, ImageDescriber
from .llm import llm_chunk_semantic, llm_summarize, llm_extract_entities, llm_translate, LLMProcessor
from .upload import upload_to_dataloop, upload_metadata_only, dry_run_upload, ChunkUploader

__all__ = [
    # Text Normalization
    'clean',
    'normalize_whitespace',
    'remove_empty_lines',
    'deep_clean',
    'TextNormalizer',
    # Chunking
    'chunk',
    'chunk_with_images',
    'TextChunker',
    # OCR
    'ocr_enhance',
    'describe_images',
    'OCREnhancer',
    'ImageDescriber',
    # LLM
    'llm_chunk_semantic',
    'llm_summarize',
    'llm_extract_entities',
    'llm_translate',
    'LLMProcessor',
    # Upload
    'upload_to_dataloop',
    'upload_metadata_only',
    'dry_run_upload',
    'ChunkUploader',
]
