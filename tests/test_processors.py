"""
Integration tests for PDF and DOC processors.

Usage:
    pytest tests/test_processors.py -v
    pytest tests/test_processors.py -k pdf
    pytest tests/test_processors.py -k doc

Edit TEST_ITEMS, PDF_CONFIG, and DOC_CONFIG in tests/test_config.py.
"""

import sys
import traceback
from unittest.mock import MagicMock

import pytest
import dtlpy as dl

from apps.pdf_processor.app import PDFProcessor
from apps.doc_processor.app import DOCProcessor
from tests.test_config import TEST_ITEMS, TARGET_DATASET_ID, PDF_CONFIG, DOC_CONFIG


# Processor registry for parameterized tests
PROCESSORS = {
    'pdf': {
        'class': PDFProcessor,
        'config': PDF_CONFIG,
        'label': 'PDF',
    },
    'doc': {
        'class': DOCProcessor,
        'config': DOC_CONFIG,
        'label': 'DOC/DOCX',
    },
}


def _create_mock_context(config: dict) -> dl.Context:
    """Create a mock dl.Context with the given config."""
    context = MagicMock(spec=dl.Context)
    context.node = MagicMock()
    context.node.metadata = {'customNodeConfig': config}
    return context


@pytest.mark.parametrize("file_type", ['pdf', 'doc'])
def test_processor(file_type):
    """
    Test processor with a Dataloop item.

    Uses TEST_ITEMS, TARGET_DATASET_ID, and configs from test_config.py.
    Asserts that chunks were successfully created.
    """
    # Validate configuration
    if file_type not in TEST_ITEMS:
        pytest.skip(f"'{file_type}' test configuration not found in TEST_ITEMS")

    if not TARGET_DATASET_ID:
        pytest.fail("TARGET_DATASET_ID is required. Set it in tests/test_config.py")

    processor_info = PROCESSORS[file_type]
    processor_cls = processor_info['class']
    config = processor_info['config']
    label = processor_info['label']

    print(f"\n{'='*60}")
    print(f"Testing {label} Processor")
    print(f"{'='*60}\n")

    # Get the item
    item_id = TEST_ITEMS[file_type]['item_id']
    print(f"Fetching item: {item_id}")
    item = dl.items.get(item_id=item_id)
    print(f"Retrieved: {item.name} ({item.mimetype})")
    print(f"Source dataset: {item.dataset.name} (ID: {item.dataset.id})")

    # Get target dataset
    print(f"\nFetching target dataset: {TARGET_DATASET_ID}")
    target_dataset = dl.datasets.get(dataset_id=TARGET_DATASET_ID)
    print(f"Target dataset: {target_dataset.name} (ID: {target_dataset.id})")

    # Show configuration
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run processor
    print(f"\nProcessing {label}...")
    context = _create_mock_context(config)
    chunk_items = processor_cls.run(item, target_dataset, context)

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
    # Allow running specific processor from command line
    file_type = sys.argv[1] if len(sys.argv) > 1 else 'pdf'

    if file_type not in PROCESSORS:
        print(f"Unknown file type: {file_type}")
        print(f"Available: {', '.join(PROCESSORS.keys())}")
        sys.exit(1)

    try:
        test_processor(file_type)
        print(f"\nTest completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
        sys.exit(1)
