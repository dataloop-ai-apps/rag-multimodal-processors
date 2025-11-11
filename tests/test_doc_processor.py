"""
Simple test for DOC Processor.

Usage:
    pytest tests/test_doc_processor.py
    or
    pytest tests

    Edit TEST_ITEMS['doc'] and DOC_CONFIG in tests/test_config.py, then run the script.
"""

import sys
import dtlpy as dl
from apps.doc_processor.doc_processor import DOCProcessor
from tests.test_config import TEST_ITEMS, DOC_CONFIG as CONFIG

# Get test items for DOC tests
if 'doc' not in TEST_ITEMS:
    raise ValueError("'doc' test configuration not found in TEST_ITEMS. Please add it to tests/test_config.py")
if 'dataset' not in TEST_ITEMS:
    raise ValueError("'dataset' entry not found in TEST_ITEMS. Please add it to tests/test_config.py")
ITEM_ID = TEST_ITEMS['doc']['item_id']
TARGET_DATASET_ID = TEST_ITEMS['dataset']['dataset_id']


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


def test_doc_processor():
    """
    Test DOC processor with a Dataloop item.

    Uses ITEM_ID, TARGET_DATASET_ID, and CONFIG from module-level configuration.
    Asserts that chunks were successfully created.
    """
    item_id = ITEM_ID
    target_dataset_id = TARGET_DATASET_ID
    config = CONFIG
    print(f"\n{'='*60}")
    print(f"Testing DOC Processor")
    print(f"{'='*60}\n")

    # Get the item
    print(f"📥 Fetching item: {item_id}")
    try:
        item = dl.items.get(item_id=item_id)
        print(f"✅ Retrieved: {item.name} ({item.mimetype})")
    except Exception as e:
        print(f"❌ Failed to retrieve item: {str(e)}")
        raise

    # Get or create target dataset
    print(f"\n📦 Setting up target dataset...")
    if target_dataset_id:
        try:
            target_dataset = dl.datasets.get(dataset_id=target_dataset_id)
            print(f"✅ Using specified dataset: {target_dataset.name} (ID: {target_dataset.id})")
        except Exception as e:
            print(f"❌ Failed to get target dataset: {str(e)}")
            raise
    else:
        # Auto-create chunks dataset
        original_dataset_name = item.dataset.name
        new_dataset_name = f"{original_dataset_name}_chunks"
        print(f"📝 Auto-creating dataset: {new_dataset_name}")

        try:
            # Try to get existing dataset
            target_dataset = item.project.datasets.get(dataset_name=new_dataset_name)
            print(f"✅ Using existing chunks dataset: {target_dataset.name} (ID: {target_dataset.id})")
        except dl.exceptions.NotFound:
            # Create new dataset
            target_dataset = item.project.datasets.create(dataset_name=new_dataset_name)
            print(f"✅ Created new dataset: {target_dataset.name} (ID: {target_dataset.id})")

    # Show configuration
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")

    # Create mock context
    context = MockContext(config)

    # Initialize DOC Processor
    print(f"\n🔧 Initializing DOC Processor...")
    processor = DOCProcessor()

    # Process the item
    print(f"\n⚙️  Processing DOC/DOCX...")
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
        import traceback

        traceback.print_exc()
        raise


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    """Run the test with the configured item ID, target dataset, and config."""
    # Run test
    try:
        test_doc_processor()
        print(f"\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed!")
        sys.exit(1)
