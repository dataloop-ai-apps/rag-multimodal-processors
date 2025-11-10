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
    "dataset": {'dataset_id': "6910ba261a0566b56d15a55a"},  # Shared target dataset for all test chunks
    'pdf': {'item_id': "6911a710d4c1299c6780c14f"},  # PDF file item ID
    'doc': {'item_id': "6910ba43732d419b5d98b41c"},  # DOC/DOCX file item ID
    # Add more test types as needed:
    # 'html': {
    #     'item_id': "...",  # HTML file item ID
    # },
    # 'email': {
    #     'item_id': "...",  # Email file item ID
    # },
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
    # OCR Processing
    'use_ocr': False,  # Set to True to enable OCR for scanned PDFs
    'ocr_integration_method': 'append',  # Options: 'append', 'prepend', 'separate'
    # Text Extraction
    'use_markdown_extraction': False,  # Use markdown extraction for PDFs
    'extract_images': True,
    'extract_tables': True,
    # Chunking Strategy
    'chunking_strategy': 'recursive',  # Options: 'recursive', 'fixed-size', 'sentence', 'paragraph', '1-chunk'
    'max_chunk_size': 300,
    'chunk_overlap': 20,
}

# ============================================================
# DOC TEST CONFIGURATION
# ============================================================

DOC_CONFIG = {
    # Text Extraction
    'extract_images': True,
    'extract_tables': True,
    # Chunking Strategy
    'chunking_strategy': 'recursive',  # Options: 'recursive', 'fixed-size', 'sentence', 'paragraph', '1-chunk'
    'max_chunk_size': 300,
    'chunk_overlap': 20,
}
