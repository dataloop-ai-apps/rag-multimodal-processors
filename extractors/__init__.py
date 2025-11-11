"""
Extractors package for extracting content from items.
Handles PDF, DOCX, OCR, and other extraction methods.
"""

# Import data structures
from .content_types import (
    ExtractedContent,
    ImageContent,
    TableContent,
)

# Import specific extractors
from .pdf_extractor import PDFExtractor
from .docs_extractor import DocsExtractor
from .ocr_extractor import OCRExtractor

# Import registry functions
from .registry import (
    EXTRACTOR_REGISTRY,
    get_extractor,
    register_extractor,
    get_supported_types,
)

__all__ = [
    # Data models
    'ExtractedContent',
    'ImageContent',
    'TableContent',
    # Extractors
    'PDFExtractor',
    'DocsExtractor',
    'OCRExtractor',
    # Registry
    'EXTRACTOR_REGISTRY',
    'get_extractor',
    'register_extractor',
    'get_supported_types',
]
