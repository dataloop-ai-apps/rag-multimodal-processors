# Tests

## Quick Start

```bash
# Unit tests (no Dataloop connection needed)
pytest tests/test_core.py tests/test_extractors.py tests/test_transforms.py -v

# Integration tests (requires Dataloop auth)
pytest tests/test_processors.py -v
```

## Test Files

| File | Type | Description |
|------|------|-------------|
| `test_core.py` | Unit | Config, ErrorTracker, ExtractedData, ChunkMetadata |
| `test_extractors.py` | Unit | PDFExtractor, DOCExtractor |
| `test_transforms.py` | Unit | clean, chunk, deep_clean, llm transforms |
| `test_processors.py` | Integration | PDF and DOC processor end-to-end |
| `test_config.py` | Config | Test item IDs, dataset IDs, processor settings |

## Integration Test Setup

Edit `test_config.py` with your Dataloop IDs:

```python
TEST_ITEMS = {
    'pdf': {'item_id': 'your-pdf-item-id'},
    'doc': {'item_id': 'your-docx-item-id'},
}

TARGET_DATASET_ID = 'your-target-dataset-id'

PDF_CONFIG = {
    'max_chunk_size': 500,
    'chunking_strategy': 'recursive',
    'remote_path': '/chunks',  # Upload directory
}

DOC_CONFIG = {
    'max_chunk_size': 500,
    'chunking_strategy': 'recursive',
    'remote_path': '/chunks',
}
```

Run specific processor:
```bash
pytest tests/test_processors.py -k pdf -v
pytest tests/test_processors.py -k doc -v
```

## Writing Tests

Transform test:
```python
from utils.extracted_data import ExtractedData
from utils.config import Config
import transforms

def test_my_transform():
    data = ExtractedData(config=Config())
    data.content_text = "Test content"

    result = transforms.my_transform(data)

    assert isinstance(result, ExtractedData)
```

Extractor test:
```python
from utils.extracted_data import ExtractedData
from apps.pdf_processor.pdf_extractor import PDFExtractor

def test_extractor():
    data = ExtractedData()
    result = PDFExtractor.extract(data)
    assert result.current_stage == "extraction"
```
