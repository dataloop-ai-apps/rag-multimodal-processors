"""
Apps package for file-type specific processors.

Each app processes a specific file type using shared extractors and stages.
Each app is a separate Dataloop application with its own Dockerfile and configuration.
"""

from .pdf_processor.app import PDFProcessor
from .pdf_processor.pdf_extractor import PDFExtractor
from .doc_processor.app import DOCProcessor
from .doc_processor.doc_extractor import DOCExtractor

__all__ = [
    'PDFProcessor',
    'PDFExtractor',
    'DOCProcessor',
    'DOCExtractor',
]
