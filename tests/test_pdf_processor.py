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
from unittest.mock import MagicMock

import dtlpy as dl
from apps.pdf_processor.app import PDFProcessor
from tests.test_config import TEST_ITEMS, TARGET_DATASET_ID, PDF_CONFIG as CONFIG

# Get test item configuration
if 'pdf' not in TEST_ITEMS:
    raise ValueError("'pdf' test configuration not found in TEST_ITEMS. Please add it to tests/test_config.py")

ITEM_ID = TEST_ITEMS['pdf']['item_id']


def _create_mock_context(config: dict) -> dl.Context:
    """Create a mock dl.Context with the given config."""
    context = MagicMock(spec=dl.Context)
    context.node = MagicMock()
    context.node.metadata = {'customNodeConfig': config}
    return context


def test_pdf_processor():
    """
    Test PDF processor with a Dataloop item.

    Uses ITEM_ID, TARGET_DATASET_ID, and CONFIG from module-level configuration.
    Asserts that chunks were successfully created.
    """
    print(f"\n{'='*60}")
    print(f"Testing PDF Processor")
    print(f"{'='*60}\n")

    # Get the item
    print(f"Fetching item: {ITEM_ID}")
    item = dl.items.get(item_id=ITEM_ID)
    print(f"Retrieved: {item.name} ({item.mimetype})")
    print(f"Source dataset: {item.dataset.name} (ID: {item.dataset.id})")

    # Get target dataset
    if not TARGET_DATASET_ID:
        raise ValueError("TARGET_DATASET_ID is required. Please set it in tests/test_config.py")

    print(f"\nFetching target dataset: {TARGET_DATASET_ID}")
    target_dataset = dl.datasets.get(dataset_id=TARGET_DATASET_ID)
    print(f"Target dataset: {target_dataset.name} (ID: {target_dataset.id})")

    # Show configuration
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Run processor
    print(f"\nProcessing PDF...")
    context = _create_mock_context(CONFIG)
    chunk_items = PDFProcessor.run(item, target_dataset, context)

    # Results
    print(f"\n{'='*60}")
    print(f"Success!")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(f"  Total chunks: {len(chunk_items)}")
    print(f"  Target dataset: {chunk_items[0].dataset.name if chunk_items else 'N/A'}")
    print(f"  Remote path: {chunk_items[0].dir if chunk_items else 'N/A'}")

    if chunk_items:
        print(f"\nSample chunks:")
        for i, chunk_item in enumerate(chunk_items[:3]):
            print(f"  {i+1}. {chunk_item.name}")
        if len(chunk_items) > 3:
            print(f"  ... and {len(chunk_items) - 3} more")

    assert chunk_items, "No chunks were created"
    assert len(chunk_items) > 0, "Expected at least one chunk"


if __name__ == '__main__':
    try:
        test_pdf_processor()
        print(f"\nTest completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
        sys.exit(1)
