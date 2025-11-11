"""
Extractor registry for mapping MIME types to extractor classes.
"""

from .pdf_extractor import PDFExtractor
from .docs_extractor import DocsExtractor


# Registry mapping MIME types to extractor classes
EXTRACTOR_REGISTRY = {
    # PDF
    'application/pdf': PDFExtractor,
    # Google Docs / Word
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocsExtractor,
    'application/vnd.google-apps.document': DocsExtractor,
}


def get_extractor(mime_type: str):
    """
    Get extractor for MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        Extractor instance

    Raises:
        ValueError: If MIME type not supported
    """
    extractor_class = EXTRACTOR_REGISTRY.get(mime_type)

    if not extractor_class:
        raise ValueError(f"Unsupported MIME type: {mime_type}\n" f"Supported types: {list(EXTRACTOR_REGISTRY.keys())}")

    return extractor_class()


def register_extractor(mime_type: str, extractor_class):
    """Register custom extractor"""
    EXTRACTOR_REGISTRY[mime_type] = extractor_class


def get_supported_types():
    """Get list of supported MIME types"""
    return list(EXTRACTOR_REGISTRY.keys())

