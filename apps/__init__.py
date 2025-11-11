"""
Apps package for file-type specific processors.

Each app processes a specific file type using shared extractors and stages.
Each app is a separate Dataloop application with its own Dockerfile and configuration.
"""

from .pdf_processor.pdf_processor import PDFProcessor
from .doc_processor.doc_processor import DOCProcessor

__all__ = ['PDFProcessor', 'DOCProcessor']
