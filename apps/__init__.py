"""
Apps package for file-type specific processors.

Each app processes a specific file type using shared extractors and stages.
Each app is a separate Dataloop application with its own Dockerfile and configuration.
"""

__all__ = [
    'PDFProcessor',
    'PDFExtractor',
    'DOCProcessor',
    'DOCExtractor',
    'XLSProcessor',
    'XLSExtractor',
]
