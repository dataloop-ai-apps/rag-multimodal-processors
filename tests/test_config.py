"""
Test configuration file for RAG Multimodal Processors tests.

Edit the values below to configure your test runs.
"""

# ============================================================
# TEST ITEMS AND DATASETS
# ============================================================
# Organize test items by test type/app
# The 'dataset' entry contains the shared target dataset ID
# Each test type entry (pdf, doc, etc.) contains the item ID for that test

TEST_ITEMS = {
    "dataset": {'dataset_id': "691344d0a235b51330ed5951"},  # Dataloop Demo 2025 - rag preprorcess testing
    'pdf': {'item_id': "6913452e732d419b5da2dc9c"},  # PDF file item ID
    'doc': {'item_id': "6913452dd4c1299c678a452a"},  # DOC/DOCX file item ID
}


# Convenience accessors (for backward compatibility and easier access)
def get_test_item(test_type: str) -> str:
    """Get item ID for a specific test type."""
    if test_type not in TEST_ITEMS:
        raise ValueError(f"Test type '{test_type}' not found in TEST_ITEMS. Available: {list(TEST_ITEMS.keys())}")
    return TEST_ITEMS[test_type].get('item_id', "item_id_to_test")


def get_test_dataset() -> str:
    """Get dataset ID from the shared 'dataset' entry."""
    if 'dataset' not in TEST_ITEMS:
        raise ValueError("'dataset' entry not found in TEST_ITEMS. Please add it to tests/test_config.py")
    return TEST_ITEMS['dataset'].get('dataset_id', "dataset_id_to_store_chunks")


# Legacy support (deprecated - use TEST_ITEMS instead)
ITEM_ID = TEST_ITEMS.get('pdf', {}).get('item_id', "item_id_to_test")
TARGET_DATASET_ID = TEST_ITEMS.get('dataset', {}).get('dataset_id', "dataset_id_to_store_chunks")

# ============================================================
# PDF TEST CONFIGURATION
# ============================================================

PDF_CONFIG = {
    'name': 'Test-PDF-Processor',
    # OCR Processing
    'ocr_from_images': True,  # Extract images and apply OCR (uses EasyOCR)
    'ocr_integration_method': 'separate_chunks',  # Options: 'append_to_page', 'separate_chunks', 'combine_all'
    # Text Extraction
    'use_markdown_extraction': False,
    # Chunking Strategy
    'chunking_strategy': (
        'recursive'
    ),  # Options: 'recursive', 'fixed-size', 'nltk-sentence', 'nltk-paragraphs', '1-chunk'
    'max_chunk_size': 500,
    'chunk_overlap': 20,
    # Text Cleaning
    'to_correct_spelling': True,
}

# ============================================================
# DOC TEST CONFIGURATION
# ============================================================

DOC_CONFIG = {
    'name': 'Test-DOC-Processor',
    # Text Extraction
    'extract_images': True,
    'extract_tables': True,
    # Chunking Strategy
    'chunking_strategy': (
        'recursive'
    ),  # Options: 'recursive', 'fixed-size', 'nltk-sentence', 'nltk-paragraphs', '1-chunk'
    'max_chunk_size': 500,
    'chunk_overlap': 20,
    # Text Cleaning
    'to_correct_spelling': True,
}
