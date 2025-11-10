"""
Main entry point for document processing.
Provides simple API for processing PDF and DOC files.

Supported file types:
    - PDF (.pdf)
    - Microsoft Word (.docx)

Example:
    >>> import dtlpy as dl
    >>> from main import process_pdf, process_doc
    >>>
    >>> # Process PDF
    >>> item = dl.items.get(item_id='abc123')
    >>> dataset = dl.datasets.get(dataset_id='xyz789')
    >>> chunks = process_pdf(item, dataset, use_ocr=True, max_chunk_size=500)
    >>>
    >>> # Process DOCX
    >>> chunks = process_doc(item, dataset, max_chunk_size=1000)
"""

import logging
from typing import Dict, Any, List
import dtlpy as dl

# Import app processors
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apps', 'pdf-processor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apps', 'doc-processor'))

from pdf_app import PDFApp
from doc_app import DOCApp

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# APP REGISTRY
# ============================================================================

# Map MIME types to app classes
APP_REGISTRY = {
    'application/pdf': PDFApp,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCApp,
    'application/vnd.google-apps.document': DOCApp,  # Google Docs exported as DOCX
}


def get_app_for_item(item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any]):
    """
    Get the appropriate app for a given item based on MIME type.

    Args:
        item: Dataloop item to process
        target_dataset: Target dataset for chunks
        config: Processing configuration

    Returns:
        Initialized app instance (PDFApp or DOCApp)

    Raises:
        ValueError: If file type not supported
    """
    mime_type = item.mimetype
    app_class = APP_REGISTRY.get(mime_type)

    if app_class is None:
        supported = ', '.join(APP_REGISTRY.keys())
        raise ValueError(f"Unsupported file type: {mime_type}\n" f"Supported types: {supported}")

    logger.debug(f"Using {app_class.__name__} for {mime_type}")
    return app_class(item, target_dataset, config)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================


def process_item(item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any] = None) -> List[dl.Item]:
    """
    Process any supported file type (PDF or DOC).

    Auto-detects file type from MIME type and routes to appropriate app.

    Args:
        item: Dataloop item to process
        target_dataset: Target dataset for chunks
        config: Processing configuration dict

    Returns:
        List of uploaded chunk items

    Raises:
        ValueError: If file type not supported

    Example:
        >>> item = dl.items.get(item_id='abc123')
        >>> dataset = dl.datasets.get(dataset_id='xyz789')
        >>> chunks = process_item(item, dataset, {
        ...     'use_ocr': True,
        ...     'max_chunk_size': 500
        ... })
    """
    config = config or {}
    logger.info(f"Processing {item.name} ({item.mimetype})")

    try:
        app = get_app_for_item(item, target_dataset, config)
        result = app.run()
        logger.info(f"Successfully processed {item.name}: {len(result)} chunks")
        return result

    except ValueError as e:
        logger.error(f"Unsupported file type: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Processing failed for {item.name}: {str(e)}", exc_info=True)
        raise


# ============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC FILE TYPES
# ============================================================================


def process_pdf(item: dl.Item, target_dataset: dl.Dataset, **config) -> List[dl.Item]:
    """
    Process PDF document.

    Args:
        item: PDF file item
        target_dataset: Target dataset for chunks
        **config: Configuration options:
            - use_ocr (bool): Apply OCR to images (default: False)
            - extract_images (bool): Extract images from PDF (default: True)
            - upload_images (bool): Upload extracted images to Dataloop (default: True)
            - link_images_to_chunks (bool): Associate images with chunks by page number (default: True)
            - embed_images_in_chunks (bool): Embed image references in chunk text (default: False)
            - image_marker_format (str): Format for embedded image markers: 'markdown', 'reference', or 'inline' (default: 'markdown')
            - image_context_before (int): Characters of text before image to include (default: 200)
            - image_context_after (int): Characters of text after image to include (default: 200)
            - image_upload_path (str): Remote path for uploaded images (default: '/images')
            - max_chunk_size (int): Maximum chunk size (default: 300)
            - chunk_overlap (int): Overlap between chunks (default: 20)
            - chunking_strategy (str): 'recursive', 'semantic', 'sentence', or 'paragraph'
            - llm_model_id (str): Required for semantic chunking
            - log_level (str): 'DEBUG', 'INFO', 'WARNING', 'ERROR'

    Returns:
        List of uploaded chunk items

    Example:
        >>> # Basic processing (with images)
        >>> chunks = process_pdf(item, dataset)
        >>>
        >>> # With OCR
        >>> chunks = process_pdf(item, dataset, use_ocr=True)
        >>>
        >>> # Without image extraction
        >>> chunks = process_pdf(item, dataset, extract_images=False)
        >>>
        >>> # Extract but don't upload images
        >>> chunks = process_pdf(item, dataset, upload_images=False)
        >>>
        >>> # Embed images in chunk text (multimodal chunks)
        >>> chunks = process_pdf(item, dataset, embed_images_in_chunks=True)
        >>>
        >>> # Custom chunk size
        >>> chunks = process_pdf(item, dataset, max_chunk_size=500, chunk_overlap=50)
        >>>
        >>> # Semantic chunking
        >>> chunks = process_pdf(item, dataset,
        ...                      chunking_strategy='semantic',
        ...                      llm_model_id='model-id')
    """
    return process_item(item, target_dataset, config)


def process_doc(item: dl.Item, target_dataset: dl.Dataset, **config) -> List[dl.Item]:
    """
    Process Microsoft Word document (.docx).

    Args:
        item: DOCX file item
        target_dataset: Target dataset for chunks
        **config: Configuration options:
            - max_chunk_size (int): Maximum chunk size (default: 300)
            - chunk_overlap (int): Overlap between chunks (default: 20)
            - chunking_strategy (str): 'recursive', 'semantic', 'sentence', or 'paragraph'
            - llm_model_id (str): Required for semantic chunking
            - log_level (str): 'DEBUG', 'INFO', 'WARNING', 'ERROR'

    Returns:
        List of uploaded chunk items

    Example:
        >>> # Basic processing
        >>> chunks = process_doc(item, dataset)
        >>>
        >>> # Custom chunk size
        >>> chunks = process_doc(item, dataset, max_chunk_size=1000)
        >>>
        >>> # Semantic chunking
        >>> chunks = process_doc(item, dataset,
        ...                      chunking_strategy='semantic',
        ...                      llm_model_id='model-id')
    """
    return process_item(item, target_dataset, config)


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def process_batch(
    items: List[dl.Item], target_dataset: dl.Dataset, config: Dict[str, Any] = None
) -> Dict[str, List[dl.Item]]:
    """
    Process multiple items in batch.

    Args:
        items: List of Dataloop items (PDF or DOC)
        target_dataset: Target dataset for chunks
        config: Processing configuration dict

    Returns:
        Dictionary mapping item IDs to uploaded chunk items

    Example:
        >>> items = dataset.items.list()
        >>> results = process_batch(items, target_dataset, {'use_ocr': True})
        >>> for item_id, chunks in results.items():
        ...     print(f"{item_id}: {len(chunks)} chunks")
    """
    config = config or {}
    results = {}

    logger.info(f"Starting batch processing: {len(items)} items")

    for i, item in enumerate(items, 1):
        logger.info(f"[{i}/{len(items)}] Processing {item.name}")

        try:
            uploaded = process_item(item, target_dataset, config)
            results[item.id] = uploaded
        except ValueError as e:
            logger.warning(f"Skipping {item.name}: {e}")
            results[item.id] = []
        except Exception as e:
            logger.error(f"Error processing {item.name}: {e}", exc_info=True)
            results[item.id] = []

    total_chunks = sum(len(chunks) for chunks in results.values())
    logger.info(f"Batch processing complete: {len(results)} items, {total_chunks} chunks")

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_supported_file_types() -> List[str]:
    """
    Get list of supported MIME types.

    Returns:
        List of supported MIME types

    Example:
        >>> types = get_supported_file_types()
        >>> for mime_type in types:
        ...     print(mime_type)
    """
    return list(APP_REGISTRY.keys())


if __name__ == '__main__':
    print("=== RAG Document Processor ===\n")

    print("Supported file types:")
    for mime_type in get_supported_file_types():
        print(f"  {mime_type}")

    print("\n=== Example Usage ===")
    print(
        """
    import dtlpy as dl
    from main import process_pdf, process_doc, process_batch

    # Get item and dataset
    item = dl.items.get(item_id='your-item-id')
    dataset = dl.datasets.get(dataset_id='your-dataset-id')

    # Basic PDF processing
    chunks = process_pdf(item, dataset)

    # PDF with OCR
    chunks = process_pdf(item, dataset, use_ocr=True)

    # PDF with custom chunk size
    chunks = process_pdf(item, dataset, max_chunk_size=500, chunk_overlap=50)

    # PDF with semantic chunking
    chunks = process_pdf(
        item, dataset,
        chunking_strategy='semantic',
        llm_model_id='your-model-id'
    )

    # Process DOC file
    chunks = process_doc(item, dataset, max_chunk_size=1000)

    # Batch processing
    items = dataset.items.list()
    results = process_batch(items, target_dataset, {'use_ocr': True})

    # Debug mode
    chunks = process_pdf(item, dataset, log_level='DEBUG')
    """
    )

    print("Running real example...")

    # Get items
    item = dl.items.get(
        item_id='6911a710d4c1299c6780c14f'
    )  # doc: 6910ba43732d419b5d98b41c, pdf: 6910ba43732d419b5d98b41c
    dataset = item.dataset

    # Auto-detect file type from metadata and process accordingly
    chunks = process_item(
        item, dataset, {'max_chunk_size': 1000, 'use_ocr': False}  # Set to True for PDFs that need OCR
    )
    print(f"Created {len(chunks)} chunks")
