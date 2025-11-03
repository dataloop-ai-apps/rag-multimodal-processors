"""
Main processor factory for creating processors based on MIME type.
"""

from typing import Dict, Type, Optional
from pipeline.base.processor import BaseProcessor
from apps.text_processor.text_processor import TextProcessor
from apps.html_processor.html_processor import HTMLProcessor
from apps.email_processor.email_processor import EmailProcessor
from apps.pdf_processor.pdf_processor_new import PDFProcessor


class ProcessorFactory:
    """Factory for creating processors based on MIME type."""

    # Registry of available processors
    _processors: Dict[str, Type[BaseProcessor]] = {
        'text/plain': TextProcessor,
        'text/csv': TextProcessor,
        'text/markdown': TextProcessor,
        'text/html': HTMLProcessor,
        'application/pdf': PDFProcessor,
        'message/rfc822': EmailProcessor,  # .eml files
    }

    @classmethod
    def create_processor(cls, mime_type: str) -> Optional[BaseProcessor]:
        """
        Create a processor for the given MIME type.

        Args:
            mime_type: MIME type of the file

        Returns:
            Processor instance or None if not supported
        """
        processor_class = cls._processors.get(mime_type)
        if processor_class:
            return processor_class()
        return None

    @classmethod
    def get_supported_mime_types(cls) -> list[str]:
        """
        Get list of supported MIME types.

        Returns:
            List of supported MIME types
        """
        return list(cls._processors.keys())

    @classmethod
    def is_supported(cls, mime_type: str) -> bool:
        """
        Check if a MIME type is supported.

        Args:
            mime_type: MIME type to check

        Returns:
            True if supported, False otherwise
        """
        return mime_type in cls._processors

    @classmethod
    def register_processor(cls, mime_type: str, processor_class: Type[BaseProcessor]):
        """
        Register a new processor for a MIME type.

        Args:
            mime_type: MIME type to register
            processor_class: Processor class to register
        """
        cls._processors[mime_type] = processor_class

    @classmethod
    def get_processor_info(cls) -> Dict[str, str]:
        """
        Get information about available processors.

        Returns:
            Dictionary mapping MIME types to processor descriptions
        """
        info = {}
        for mime_type, processor_class in cls._processors.items():
            info[mime_type] = processor_class.__doc__ or f"Processor for {mime_type}"
        return info


def get_processor_for_file(file_path: str) -> Optional[BaseProcessor]:
    """
    Get appropriate processor for a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Processor instance or None if not supported
    """
    import mimetypes

    # Guess MIME type from file extension
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        return ProcessorFactory.create_processor(mime_type)

    # Fallback to extension-based detection
    import os

    extension = os.path.splitext(file_path)[1].lower()

    extension_mapping = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.csv': 'text/csv',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.pdf': 'application/pdf',
        '.eml': 'message/rfc822',
    }

    mime_type = extension_mapping.get(extension)
    if mime_type:
        return ProcessorFactory.create_processor(mime_type)

    return None


def process_document(item, target_dataset, context):
    """
    Process a document using the appropriate processor.

    Args:
        item: Dataloop item to process
        target_dataset: Target dataset for chunks
        context: Processing context

    Returns:
        List of processed chunk items
    """
    # Get processor based on MIME type
    processor = ProcessorFactory.create_processor(item.mimetype)

    if not processor:
        # Try fallback to file extension
        processor = get_processor_for_file(item.name)

    if not processor:
        raise ValueError(f"No processor available for MIME type: {item.mimetype}")

    # Process the document
    return processor.process_document(item, target_dataset, context)


