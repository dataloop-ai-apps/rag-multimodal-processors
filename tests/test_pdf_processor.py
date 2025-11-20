"""
Simple test for PDF Processor.

Usage:
    pytest tests/test_pdf_processor.py
    or
    pytest tests

    Edit TEST_ITEMS['pdf'] and PDF_CONFIG in tests/test_config.py, then run the script.
"""

import sys
import traceback
import dtlpy as dl
from apps.pdf_processor.pdf_processor import PDFProcessor
from tests.test_config import TEST_ITEMS, DESTINATION_DATASET_ID, PDF_CONFIG as CONFIG

# Get test item configuration
if 'pdf' not in TEST_ITEMS:
    raise ValueError("'pdf' test configuration not found in TEST_ITEMS. Please add it to tests/test_config.py")

ITEM_ID = TEST_ITEMS['pdf']['item_id']
TARGET_DATASET_ID = DESTINATION_DATASET_ID


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


def test_pdf_processor():
    """
    Test PDF processor with a Dataloop item.

    Uses ITEM_ID, TARGET_DATASET_ID, and CONFIG from module-level configuration.
    Asserts that chunks were successfully created.
    """
    item_id = ITEM_ID
    target_dataset_id = TARGET_DATASET_ID
    config = CONFIG
    print(f"\n{'='*60}")
    print(f"Testing PDF Processor")
    print(f"{'='*60}\n")

    # Get the item
    print(f"📥 Fetching item: {item_id}")
    try:
        item = dl.items.get(item_id=item_id)
        print(f"✅ Retrieved: {item.name} ({item.mimetype})")
        print(f"   Source dataset: {item.dataset.name} (ID: {item.dataset.id})")
    except Exception as e:
        print(f"❌ Failed to retrieve item: {str(e)}")
        raise

    # Get destination dataset (REQUIRED)
    print(f"\n📦 Setting up destination dataset...")
    if not target_dataset_id:
        raise ValueError("DESTINATION_DATASET_ID is required. Please set it in tests/test_config.py")

    try:
        target_dataset = dl.datasets.get(dataset_id=target_dataset_id)
        print(f"✅ Using destination dataset: {target_dataset.name} (ID: {target_dataset.id})")
    except Exception as e:
        print(f"❌ Failed to get destination dataset: {str(e)}")
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
        chunk_items = processor.process_document(item, target_dataset, context)

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

        # Assert we got results
        assert chunk_items, "No chunks were created"
        assert len(chunk_items) > 0, "Expected at least one chunk"

    except Exception as e:
        print(f"\n❌ Failed: {str(e)}")
        traceback.print_exc()
        raise


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    """Run the test with the configured item ID, target dataset, and config."""
    # Run test
    try:
        test_pdf_processor()
        print(f"\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed!")
        sys.exit(1)
