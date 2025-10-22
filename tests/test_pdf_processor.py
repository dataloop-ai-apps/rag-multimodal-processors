"""
Simple test for PDF Processor.

Usage:
    python tests/test_pdf_processor.py
    
    Edit the CONFIG and ITEM_ID variables below, then run the script.
"""

import sys
import os

# Add repository root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dtlpy as dl

# Import from apps folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../apps/pdf-processor')))
from pdf_processor import PDFProcessor


# ============================================================
# CONFIGURATION - Edit these values
# ============================================================

# Your Dataloop item ID to test
ITEM_ID = "item_id"

# Configuration (matches dataloop.json schema)
CONFIG = {
    'name': 'Test-PDF-Processor',
    
    # OCR Processing
    'ocr_from_images': False,  # Extract images and apply OCR
    'custom_ocr_model_id': None,  # Leave as None to use EasyOCR (default), or provide a deployed Dataloop OCR model ID
    'ocr_integration_method': 'append_to_page',  # Options: 'append_to_page', 'separate_chunks', 'combine_all'
    
    # Text Extraction
    'use_markdown_extraction': False,
    
    # Chunking Strategy
    'chunking_strategy': 'recursive',  # Options: 'recursive', 'fixed-size', 'nltk-sentence', 'nltk-paragraphs', '1-chunk'
    'max_chunk_size': 5000,
    'chunk_overlap': 20,
    
    # Text Cleaning
    'to_correct_spelling': True,
    
    # Output Settings
    'remote_path_for_chunks': '/chunks',
    'target_dataset': None,  # None = auto-create {dataset_name}_chunks
}


# ============================================================
# Mock Context (replicates Dataloop pipeline context)
# ============================================================

class MockNode:
    """Mock Dataloop node with configuration."""
    def __init__(self, config):
        self.metadata = {'customNodeConfig': config}


class MockContext:
    """Mock Dataloop context with node configuration."""
    def __init__(self, config):
        self.node = MockNode(config)


# ============================================================
# Test Function
# ============================================================

def test_pdf_processor(item_id: str, config: dict):
    """
    Test PDF processor with a Dataloop item.
    
    Args:
        item_id: Dataloop item ID to process
        config: Configuration dictionary
        
    Returns:
        List of chunk items created
    """
    print(f"\n{'='*60}")
    print(f"Testing PDF Processor")
    print(f"{'='*60}\n")
    
    # Get the item
    print(f"📥 Fetching item: {item_id}")
    try:
        item = dl.items.get(item_id=item_id)
        print(f"✅ Retrieved: {item.name} ({item.mimetype})")
    except Exception as e:
        print(f"❌ Failed to retrieve item: {str(e)}")
        raise
    
    # Show configuration
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # Create mock context
    context = MockContext(config)
    
    # Initialize PDF Processor
    print(f"\n🔧 Initializing PDF Processor...")
    processor = PDFProcessor()
    
    # Process the item
    print(f"\n⚙️  Processing PDF...")
    try:
        chunk_items = processor.process_document(item, context)
        
        print(f"\n{'='*60}")
        print(f"✅ Success!")
        print(f"{'='*60}")
        print(f"\n📊 Results:")
        print(f"  • Total chunks: {len(chunk_items)}")
        print(f"  • Target dataset: {chunk_items[0].dataset.name if chunk_items else 'N/A'}")
        print(f"  • Remote path: {chunk_items[0].dir if chunk_items else 'N/A'}")
        
        if chunk_items:
            print(f"\n📝 Sample chunks:")
            for i, chunk_item in enumerate(chunk_items[:3]):
                print(f"  {i+1}. {chunk_item.name}")
            if len(chunk_items) > 3:
                print(f"  ... and {len(chunk_items) - 3} more")
        
        return chunk_items
        
    except Exception as e:
        print(f"\n❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    """Run the test with the configured item ID and config."""
    # Run test
    try:
        chunks = test_pdf_processor(ITEM_ID, CONFIG)
        print(f"\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed!")
        sys.exit(1)
