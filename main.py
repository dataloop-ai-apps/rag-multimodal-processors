"""
Main entry point for document processing.
Provides simple API for processing different file types with various processing levels.

Processing Levels:
- basic: Clean -> Chunk -> Upload
- ocr: OCR -> Clean -> Chunk -> Upload
- llm: Clean -> LLM Semantic Chunk -> Upload
- advanced: OCR -> Image Descriptions -> Clean -> LLM Chunk -> Upload

Example:
    >>> import dtlpy as dl
    >>> from main import process_item
    >>>
    >>> item = dl.items.get(item_id='abc123')
    >>> dataset = dl.datasets.get(dataset_id='xyz789')
    >>> result = process_item(item, dataset, 'ocr', {'use_ocr': True})
"""

from typing import Dict, Any, List, Callable
import dtlpy as dl

from extractors import get_extractor, ExtractedContent
import stages


# ============================================================================
# PROCESSING LEVEL IMPLEMENTATIONS
# ============================================================================

def basic_processing(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic text processing: clean -> chunk -> upload

    Args:
        data: Extracted content as dict
        config: Configuration dict

    Returns:
        Processed data with uploaded_items
    """
    # Clean text
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)

    # Chunk
    data = stages.chunk_recursive(data, config)

    # Upload
    data = stages.upload_to_dataloop(data, config)

    return data


def ocr_processing(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    OCR processing: OCR -> clean -> chunk -> upload

    Args:
        data: Extracted content with images
        config: Must have use_ocr=True

    Returns:
        Processed data with uploaded_items
    """
    # Enable OCR
    config['use_ocr'] = True

    # Add OCR text
    data = stages.ocr_enhance(data, config)

    # Clean text
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)

    # Chunk
    data = stages.chunk_recursive(data, config)

    # Upload
    data = stages.upload_to_dataloop(data, config)

    return data


def llm_processing(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM processing: clean -> LLM semantic chunk -> upload

    Args:
        data: Extracted content
        config: Must have llm_model_id

    Returns:
        Processed data with uploaded_items
    """
    # Clean text
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)

    # LLM semantic chunking
    data = stages.llm_chunk_semantic(data, config)

    # Upload
    data = stages.upload_to_dataloop(data, config)

    return data


def advanced_processing(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced processing: OCR -> image descriptions -> clean -> LLM chunk -> upload

    Args:
        data: Extracted content with images
        config: Must have use_ocr=True, llm_model_id, vision_model_id

    Returns:
        Processed data with uploaded_items
    """
    # Enable OCR and image descriptions
    config['use_ocr'] = True
    config['describe_images'] = True

    # Add OCR text
    data = stages.ocr_enhance(data, config)

    # Generate image descriptions
    data = stages.describe_images_with_dataloop(data, config)

    # Clean text
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)

    # LLM semantic chunking
    data = stages.llm_chunk_semantic(data, config)

    # Upload with images
    data = stages.upload_with_images(data, config)

    return data


# Map processing levels to functions
PROCESSING_LEVELS = {
    'basic': basic_processing,
    'ocr': ocr_processing,
    'llm': llm_processing,
    'advanced': advanced_processing,
}


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_item(
    item: dl.Item,
    target_dataset: dl.Dataset,
    processing_level: str = 'basic',
    config: Dict[str, Any] = None,
    verbose: bool = False
) -> List[dl.Item]:
    """
    Main entry point for processing any Dataloop item.

    Args:
        item: Dataloop item to process
        target_dataset: Target dataset for chunks
        processing_level: One of 'basic', 'ocr', 'llm', 'advanced'
        config: Optional configuration dict
        verbose: Print processing progress

    Returns:
        List of uploaded chunk items

    Example:
        >>> item = dl.items.get(item_id='abc123')
        >>> dataset = dl.datasets.get(dataset_id='xyz789')
        >>> result = process_item(item, dataset, 'ocr', {'max_chunk_size': 500})
    """
    config = config or {}

    # Step 1: Get extractor based on MIME type
    if verbose:
        print(f"Processing {item.name} (MIME: {item.mimetype})")

    extractor = get_extractor(item.mimetype)

    if verbose:
        print(f"Using {extractor}")

    # Step 2: Extract content (multimodal)
    if verbose:
        print("Extracting content...")

    extracted: ExtractedContent = extractor.extract(item, config)

    if verbose:
        print(f"Extracted: {len(extracted.text)} chars, "
              f"{len(extracted.images)} images, "
              f"{len(extracted.tables)} tables")

    # Step 3: Convert to dict for processing
    data = extracted.to_dict()
    data['item'] = item
    data['target_dataset'] = target_dataset

    # Step 4: Get processing function
    processing_func = PROCESSING_LEVELS.get(processing_level)
    if not processing_func:
        raise ValueError(
            f"Unknown processing level: {processing_level}. "
            f"Available: {list(PROCESSING_LEVELS.keys())}"
        )

    # Step 5: Run processing
    if verbose:
        print(f"Running {processing_level} processing...")

    result = processing_func(data, config)

    # Step 6: Return uploaded items
    uploaded_items = result.get('uploaded_items', [])

    if verbose:
        print(f"Processing complete: {len(uploaded_items)} chunks uploaded")

    return uploaded_items


# ============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC FILE TYPES
# ============================================================================

def process_text(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'basic',
    **config
) -> List[dl.Item]:
    """
    Process text document (.txt, .md, .csv).

    Args:
        item: Text file item
        target_dataset: Target dataset
        level: Processing level
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_text(item, dataset, level='basic')
    """
    return process_item(item, target_dataset, level, config)


def process_pdf(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'basic',
    **config
) -> List[dl.Item]:
    """
    Process PDF document.

    Args:
        item: PDF file item
        target_dataset: Target dataset
        level: Processing level ('basic', 'ocr', 'llm', 'advanced')
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_pdf(item, dataset, level='ocr', use_ocr=True)
    """
    return process_item(item, target_dataset, level, config)


def process_docs(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'basic',
    **config
) -> List[dl.Item]:
    """
    Process Google Docs (.docx).

    Args:
        item: .docx file item
        target_dataset: Target dataset
        level: Processing level
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_docs(item, dataset, level='basic')
    """
    return process_item(item, target_dataset, level, config)


def process_html(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'basic',
    **config
) -> List[dl.Item]:
    """
    Process HTML document.

    Args:
        item: HTML file item
        target_dataset: Target dataset
        level: Processing level
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_html(item, dataset, level='basic')
    """
    return process_item(item, target_dataset, level, config)


def process_image(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'ocr',
    **config
) -> List[dl.Item]:
    """
    Process image (requires OCR).

    Args:
        item: Image file item
        target_dataset: Target dataset
        level: Processing level (default: 'ocr')
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_image(item, dataset, level='ocr', use_ocr=True)
    """
    config['use_ocr'] = True
    return process_item(item, target_dataset, level, config)


def process_email(
    item: dl.Item,
    target_dataset: dl.Dataset,
    level: str = 'basic',
    **config
) -> List[dl.Item]:
    """
    Process email (.eml).

    Args:
        item: Email file item
        target_dataset: Target dataset
        level: Processing level
        **config: Additional configuration

    Returns:
        List of uploaded chunk items

    Example:
        >>> result = process_email(item, dataset, level='basic')
    """
    return process_item(item, target_dataset, level, config)


# ============================================================================
# CUSTOM PROCESSING
# ============================================================================

def process_custom(
    item: dl.Item,
    target_dataset: dl.Dataset,
    stage_functions: List[Callable],
    config: Dict[str, Any] = None,
    verbose: bool = False
) -> List[dl.Item]:
    """
    Process item with custom sequence of stages.

    Args:
        item: Dataloop item
        target_dataset: Target dataset
        stage_functions: List of stage functions to execute in order
        config: Configuration dict
        verbose: Print progress

    Returns:
        List of uploaded items

    Example:
        >>> custom_stages = [
        ...     stages.ocr_enhance,
        ...     stages.clean_text,
        ...     stages.chunk_by_sentence,
        ...     stages.upload_to_dataloop
        ... ]
        >>> result = process_custom(item, dataset, custom_stages, {'use_ocr': True})
    """
    config = config or {}

    # Extract content
    extractor = get_extractor(item.mimetype)
    extracted = extractor.extract(item, config)

    # Prepare data
    data = extracted.to_dict()
    data['item'] = item
    data['target_dataset'] = target_dataset

    # Execute stages sequentially
    for i, stage_func in enumerate(stage_functions):
        if verbose:
            stage_name = getattr(stage_func, '__name__', f'Stage{i}')
            print(f"Running stage {i+1}/{len(stage_functions)}: {stage_name}")

        data = stage_func(data, config)

    return data.get('uploaded_items', [])


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(
    items: List[dl.Item],
    target_dataset: dl.Dataset,
    processing_level: str = 'basic',
    config: Dict[str, Any] = None,
    verbose: bool = False
) -> Dict[str, List[dl.Item]]:
    """
    Process multiple items in batch.

    Args:
        items: List of Dataloop items
        target_dataset: Target dataset
        processing_level: Processing level
        config: Configuration dict
        verbose: Print progress

    Returns:
        Dictionary mapping item IDs to uploaded chunk items

    Example:
        >>> items = dataset.items.list()
        >>> results = process_batch(items, target_dataset, 'ocr')
    """
    config = config or {}
    results = {}

    for i, item in enumerate(items):
        if verbose:
            print(f"\n[{i+1}/{len(items)}] Processing {item.name}")

        try:
            uploaded = process_item(item, target_dataset, processing_level, config, verbose=verbose)
            results[item.id] = uploaded
        except Exception as e:
            print(f"Error processing {item.name}: {e}")
            results[item.id] = []

    if verbose:
        total_chunks = sum(len(chunks) for chunks in results.values())
        print(f"\nBatch processing complete: {len(results)} items, {total_chunks} chunks")

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_levels() -> Dict[str, str]:
    """
    Get list of available processing levels.

    Returns:
        Dictionary mapping level names to descriptions

    Example:
        >>> levels = get_available_levels()
        >>> for name, desc in levels.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        'basic': 'Basic text processing (clean -> chunk -> upload)',
        'ocr': 'OCR processing (OCR -> clean -> chunk -> upload)',
        'llm': 'LLM semantic chunking (clean -> LLM chunk -> upload)',
        'advanced': 'Full multimodal processing (OCR -> image descriptions -> LLM chunk -> upload)',
    }


def get_supported_file_types() -> List[str]:
    """
    Get list of supported MIME types.

    Returns:
        List of supported MIME types

    Example:
        >>> types = get_supported_file_types()
        >>> print(types)
    """
    from extractors import get_supported_types
    return get_supported_types()


def register_processing_level(name: str, func: Callable) -> None:
    """
    Register a custom processing level.

    Args:
        name: Level name
        func: Processing function with signature (data, config) -> data

    Example:
        >>> def my_processing(data, config):
        ...     data = stages.clean_text(data, config)
        ...     data = stages.my_custom_stage(data, config)
        ...     data = stages.upload_to_dataloop(data, config)
        ...     return data
        >>>
        >>> register_processing_level('my_level', my_processing)
        >>> result = process_item(item, dataset, 'my_level')
    """
    PROCESSING_LEVELS[name] = func


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Example usage and testing.
    """
    import dtlpy as dl

    # Setup
    dl.setenv('prod')

    print("=== RAG Multimodal Processor ===\n")

    # Show available levels
    print("Available processing levels:")
    for name, desc in get_available_levels().items():
        print(f"  {name}: {desc}")

    print("\nSupported file types:")
    for mime_type in get_supported_file_types():
        print(f"  {mime_type}")

    # Example usage (requires valid item and dataset IDs)
    print("\n=== Example Usage ===")
    print("""
    # Basic processing
    item = dl.items.get(item_id='your-item-id')
    dataset = dl.datasets.get(dataset_id='your-dataset-id')
    result = process_pdf(item, dataset, level='basic')

    # OCR processing
    result = process_pdf(item, dataset, level='ocr', use_ocr=True)

    # LLM processing
    result = process_pdf(
        item, dataset, level='llm',
        llm_model_id='your-model-id'
    )

    # Advanced processing
    result = process_pdf(
        item, dataset, level='advanced',
        use_ocr=True,
        llm_model_id='your-llm-model',
        vision_model_id='your-vision-model'
    )

    # Custom processing with nested function calls
    custom_stages = [
        stages.ocr_enhance,
        stages.clean_text,
        stages.chunk_by_sentence,
        stages.upload_to_dataloop
    ]
    result = process_custom(item, dataset, custom_stages, {'use_ocr': True})
    """)
