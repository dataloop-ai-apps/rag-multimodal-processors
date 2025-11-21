"""
Test configuration file for RAG Multimodal Processors tests.

Edit the values below to configure your test runs.
"""

# ============================================================
# TEST ITEMS AND DATASETS
# ============================================================
# Test items (source dataset is obtained from item.dataset)
TEST_ITEMS = {
    'pdf': {'item_id': "6911a710d4c1299c6780c14f"},
    'doc': {'item_id': "6910ba43732d419b5d98b41c"},
}

# Target dataset where chunks will be uploaded (REQUIRED)
TARGET_DATASET_ID = "6910ba261a0566b56d15a55a"  # Dataloop Demo 2025 - rag preprocess testing

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
    'correct_spelling': True,
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
    'correct_spelling': True,
}
