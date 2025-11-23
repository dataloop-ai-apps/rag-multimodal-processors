"""
Processing transforms for document processing.

All transforms follow signature: (data: ExtractedData) -> ExtractedData
"""

from .text_normalization import clean, normalize_whitespace, remove_empty_lines
from .chunking import chunk, chunk_with_images, TextChunker
from .ocr import ocr_enhance, describe_images, ocr_batch_enhance
from .llm import llm_chunk_semantic, llm_summarize, llm_extract_entities, llm_translate
from utils.upload import upload_to_dataloop, upload_metadata_only, dry_run_upload

__all__ = [
    # Text Normalization
    'clean',
    'normalize_whitespace',
    'remove_empty_lines',
    # Chunking
    'chunk',
    'chunk_with_images',
    'TextChunker',
    # OCR
    'ocr_enhance',
    'describe_images',
    'ocr_batch_enhance',
    # LLM
    'llm_chunk_semantic',
    'llm_summarize',
    'llm_extract_entities',
    'llm_translate',
    # Upload
    'upload_to_dataloop',
    'upload_metadata_only',
    'dry_run_upload',
]
