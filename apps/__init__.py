"""
Apps package for file-type specific processors.

Each app processes a specific file type using shared extractors and stages.
Each app is a separate Dataloop application with its own Dockerfile and configuration.
"""

from .pdf_processor.app import PDFProcessor
from .doc_processor.app import DOCProcessor

__all__ = ['PDFProcessor', 'DOCProcessor']
