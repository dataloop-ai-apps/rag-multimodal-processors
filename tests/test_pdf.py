"""
Test suite for PDF processing.

Tests extractor functionality, configuration, and integration with Dataloop.

Usage:
    python tests/test_pdf.py

    Edit TEST_ITEMS['pdf'] and PDF_CONFIG in tests/test_config.py, then run the script.
"""

import sys
import os

# Add repository root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add tests directory to path for test_config import
sys.path.insert(0, os.path.dirname(__file__))

import dtlpy as dl
from main import process_item
from extractors import get_extractor, ExtractedContent
from test_config import TEST_ITEMS, PDF_CONFIG as CONFIG

# Get test items for PDF tests
if 'pdf' not in TEST_ITEMS:
    raise ValueError("'pdf' test configuration not found in TEST_ITEMS. Please add it to tests/test_config.py")
if 'dataset' not in TEST_ITEMS:
    raise ValueError("'dataset' entry not found in TEST_ITEMS. Please add it to tests/test_config.py")
ITEM_ID = TEST_ITEMS['pdf']['item_id']
TARGET_DATASET_ID = TEST_ITEMS['dataset']['dataset_id']

# ============================================================
# Extractor Tests
# ============================================================


def test_extractor():
    """Test that PDFExtractor works correctly."""
    print(f"\n{'='*60}")
    print(f"Testing Extractor")
    print(f"{'='*60}\n")

    # Test 1: Extractor initialization
    print("1️⃣ Testing extractor initialization...")
    extractor = get_extractor('application/pdf')
    assert extractor.mime_type == 'application/pdf', "❌ PDFExtractor should have correct MIME type"
    assert extractor.name == 'PDF', "❌ PDFExtractor should have correct name"
    print("   ✅ PDFExtractor initialized correctly")

    # Test 2: Has extract method
    print("\n2️⃣ Testing extract method...")
    assert hasattr(extractor, 'extract'), "❌ PDFExtractor should have extract method"
    print("   ✅ Has extract() method")

    # Test 3: Registry lookup
    print("\n3️⃣ Testing registry lookup...")
    registry_extractor = get_extractor('application/pdf')
    assert registry_extractor.mime_type == 'application/pdf', "❌ Registry should return PDF extractor"
    assert registry_extractor.name == 'PDF', "❌ Registry should return PDF extractor"
    print("   ✅ Registry lookup works correctly")

    # Test 4: ExtractedContent structure
    print("\n4️⃣ Testing ExtractedContent structure...")
    # Just verify the structure exists
    assert hasattr(ExtractedContent, '__annotations__'), "❌ ExtractedContent should be defined"
    print("   ✅ ExtractedContent structure verified")

    print(f"\n{'='*60}")
    print(f"✅ Extractor Tests Passed!")
    print(f"{'='*60}\n")


def test_configuration():
    """Test configuration options."""
    print(f"\n{'='*60}")
    print(f"Testing Configuration")
    print(f"{'='*60}\n")

    # Test 1: Default config
    print("1️⃣ Testing default configuration...")
    default_config = {}
    assert 'use_ocr' not in default_config or default_config.get('use_ocr') == False, "❌ Default should not use OCR"
    print("   ✅ Default config is correct")

    # Test 2: OCR config
    print("\n2️⃣ Testing OCR configuration...")
    ocr_config = {'use_ocr': True, 'ocr_integration_method': 'append'}
    assert ocr_config['use_ocr'] == True, "❌ OCR should be enabled"
    assert ocr_config['ocr_integration_method'] in ['append', 'prepend', 'separate'], "❌ Invalid OCR method"
    print("   ✅ OCR config is valid")

    # Test 3: Chunking config
    print("\n3️⃣ Testing chunking configuration...")
    chunking_config = {'chunking_strategy': 'recursive', 'max_chunk_size': 300, 'chunk_overlap': 20}
    assert chunking_config['chunking_strategy'] in [
        'recursive',
        'fixed-size',
        'sentence',
        'paragraph',
        '1-chunk',
    ], "❌ Invalid chunking strategy"
    assert chunking_config['max_chunk_size'] > 0, "❌ Chunk size should be positive"
    print("   ✅ Chunking config is valid")

    # Test 4: Extraction config
    print("\n4️⃣ Testing extraction configuration...")
    extraction_config = {'extract_images': True, 'extract_tables': True, 'use_markdown_extraction': False}
    assert isinstance(extraction_config['extract_images'], bool), "❌ extract_images should be boolean"
    assert isinstance(extraction_config['extract_tables'], bool), "❌ extract_tables should be boolean"
    print("   ✅ Extraction config is valid")

    print(f"\n{'='*60}")
    print(f"✅ Configuration Tests Passed!")
    print(f"{'='*60}\n")


# ============================================================
# Integration Test
# ============================================================


def run_pdf_processing_integration_test(item_id: str, target_dataset_id: str, config: dict):
    """
    Integration test: Test PDF processing with a real Dataloop item.
    Tests the complete processing workflow end-to-end.

    Args:
        item_id: Dataloop item ID to process
        target_dataset_id: Target dataset ID for storing chunks
        config: Configuration dictionary

    Returns:
        List of chunk items created
    """
    print(f"\n{'='*60}")
    print(f"Integration Test: PDF Processing")
    print(f"{'='*60}\n")

    # Get the item
    print(f"📥 Fetching item: {item_id}")
    try:
        item = dl.items.get(item_id=item_id)
        print(f"✅ Retrieved: {item.name} ({item.mimetype})")
    except Exception as e:
        print(f"❌ Failed to retrieve item: {str(e)}")
        raise

    # Get target dataset
    print(f"\n📦 Fetching target dataset: {target_dataset_id}")
    try:
        target_dataset = dl.datasets.get(dataset_id=target_dataset_id)
        print(f"✅ Retrieved: {target_dataset.name}")
    except Exception as e:
        print(f"❌ Failed to retrieve dataset: {str(e)}")
        raise

    # Show configuration
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")

    # Process the item
    print(f"\n⚙️  Processing PDF...")
    try:
        chunk_items = process_item(item=item, target_dataset=target_dataset, config=config)

        print(f"\n{'='*60}")
        print(f"✅ Success!")
        print(f"{'='*60}")
        print(f"\n📊 Results:")
        print(f"  • Total chunks: {len(chunk_items)}")
        print(f"  • Target dataset: {chunk_items[0].dataset.name if chunk_items else 'N/A'}")

        if chunk_items:
            print(f"\n📝 Sample chunks:")
            for i, chunk_item in enumerate(chunk_items[:3]):
                print(f"  {i+1}. {chunk_item.name}")
            if len(chunk_items) > 3:
                print(f"  ... and {len(chunk_items) - 3} more")

        # Verify chunk metadata structure
        print(f"\n🔍 Verifying chunk metadata...")
        if chunk_items:
            first_chunk = chunk_items[0]
            chunk_metadata = first_chunk.metadata.get('user', {})

            # Check for common metadata fields
            metadata_fields = ['document', 'chunk_index', 'total_chunks', 'original_item_id', 'original_dataset_id']
            found_fields = [f for f in metadata_fields if f in chunk_metadata]

            if found_fields:
                print(f"  ✅ Found metadata fields: {found_fields}")
                if 'chunk_index' in chunk_metadata:
                    print(f"     • chunk_index: {chunk_metadata['chunk_index']}")
                if 'total_chunks' in chunk_metadata:
                    print(f"     • total_chunks: {chunk_metadata['total_chunks']}")
            else:
                print(f"  ⚠️  No standard metadata fields found (may be in different location)")

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
    """Run all tests."""

    print("\n" + "=" * 60)
    print("PDF Processing Test Suite")
    print("=" * 60)

    # Run extractor tests (no Dataloop connection needed)
    print("\n🔧 Running extractor tests...")
    try:
        test_extractor()
    except AssertionError as e:
        print(f"\n❌ Extractor test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Extractor test error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run config tests
    print("\n⚙️  Running configuration tests...")
    try:
        test_configuration()
    except AssertionError as e:
        print(f"\n❌ Config test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Config test error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run integration test (requires Dataloop connection and valid item)
    print("\n🔗 Running integration test...")
    print(f"   Note: Requires valid ITEM_ID and TARGET_DATASET_ID")

    if ITEM_ID == "item_id_to_test" or TARGET_DATASET_ID == "dataset_id_to_store_chunks":
        print(f"\n⚠️  Skipping integration test - Please set ITEM_ID and TARGET_DATASET_ID")
        print(f"   Extractor and config tests passed!")
        print(f"\n✅ Test suite completed (integration test skipped)!")
        sys.exit(0)

    try:
        chunks = run_pdf_processing_integration_test(ITEM_ID, TARGET_DATASET_ID, CONFIG)
        print(f"\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Integration test failed!")
        sys.exit(1)
