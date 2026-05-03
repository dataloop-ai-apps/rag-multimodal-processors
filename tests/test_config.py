"""
Test configuration file for RAG Multimodal Processors tests.

Edit the values below to configure your test runs.
"""

# ============================================================
# TEST ITEMS AND DATASETS
# ============================================================
# Test items (source dataset is obtained from item.dataset)
TEST_ITEMS = {
    'pdf': {'item_id': "69f0dd3364b5f2de3f19cb5b"},
    'doc': {'item_id': "69f0dd42c56bbd3672dc6767"},
    'pptx': {'item_id': "69f32d5a700e6fd65d1c6fec"}, 
}

# Target dataset where chunks will be uploaded (REQUIRED)
TARGET_DATASET_ID = "69f34f7e471097b6b35983b5"  # Model mgmt demo: RAG demo source

# ============================================================
# PDF TEST CONFIGURATION
# ============================================================

PDF_CONFIG = {
    'name': 'Test-PDF-Processor',
    # Extraction settings
    'extraction_method': 'basic',  # Options: 'markdown', 'basic'
    'extract_images': True,
    # OCR Processing
    'use_ocr': False,  # Enable OCR on extracted images (uses EasyOCR)
    'ocr_method': 'local',  # Options: 'local', 'batch', 'auto'
    # Chunking Strategy
    'chunking_strategy': 'recursive',  # Options: 'recursive', 'fixed', 'sentence', 'none'
    'max_chunk_size': 500,
    'chunk_overlap': 20,
    # Text Cleaning
    'normalize_whitespace': True,
    'remove_empty_lines': True,
    # Upload settings
    'remote_path': '/chunks',  # Remote directory for uploaded chunks
}

# ============================================================
# DOC TEST CONFIGURATION
# ============================================================

DOC_CONFIG = {
    'name': 'Test-DOC-Processor',
    # Extraction settings
    'extract_images': True,
    'extract_tables': True,
    # Chunking Strategy
    'chunking_strategy': 'recursive',  # Options: 'recursive', 'fixed', 'sentence', 'none'
    'max_chunk_size': 500,
    'chunk_overlap': 20,
    # Text Cleaning
    'normalize_whitespace': True,
    'remove_empty_lines': True,
    # Upload settings
    'remote_path': '/chunks',  # Remote directory for uploaded chunks
}

# ============================================================
# PPTX TEST CONFIGURATION
# ============================================================

PPTX_CONFIG = {
    'name': 'Test-PPTX-Processor',
    # Extraction settings
    'extract_images': True,
    'extract_tables': True,
    # Chunking Strategy
    'chunking_strategy': 'recursive',
    'max_chunk_size': 500,
    'chunk_overlap': 40,
    # Text Cleaning
    'to_correct_spelling': False,
    # Upload settings
    'remote_path': '/chunks',
}
