# Tests

Comprehensive test suite for RAG Multimodal Processors with 108 unit tests.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_transforms.py -v
pytest tests/test_extracted_data.py -v
pytest tests/test_extractors.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Test Structure

### Unit Tests (108 total)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_utils_config.py` | 16 | Config dataclass and validation |
| `test_utils_errors.py` | 20 | ErrorTracker class |
| `test_extracted_data.py` | 24 | ExtractedData dataclass |
| `test_extractors.py` | 16 | PDFExtractor and DOCExtractor |
| `test_transforms.py` | 32 | All transform functions |

### Test Categories

**Config Tests** (`test_utils_config.py`):
- Default configuration values
- Config creation from dict
- Validation of chunk sizes, overlap
- OCR configuration validation
- Error mode validation

**Error Tracker Tests** (`test_utils_errors.py`):
- Error and warning recording
- Stop mode vs continue mode
- Max errors threshold
- Error summaries

**ExtractedData Tests** (`test_extracted_data.py`):
- Dataclass creation and defaults
- Error logging through `log_error()` and `log_warning()`
- Content retrieval (`get_text()`, `has_content()`)
- Pipeline stage tracking
- Summary generation

**Extractor Tests** (`test_extractors.py`):
- PDF extraction with PyMuPDF
- DOCX extraction with python-docx
- Error handling for missing items
- Metadata population
- Table to markdown conversion

**Transform Tests** (`test_transforms.py`):
- Text cleaning transforms
- Whitespace normalization
- Chunking with various strategies
- Chunking with image association
- LLM transforms (without model)
- Transform signature verification
- Transform chaining

### Integration Tests

| Test File | Description |
|-----------|-------------|
| `test_pdf_processor.py` | PDF processing integration |
| `test_doc_processor.py` | DOCX processing integration |
| `test_static_methods.py` | Static method architecture |
| `test_chunk_metadata.py` | ChunkMetadata dataclass |
| `test_data_types.py` | Data type classes |

## Configuration

Integration tests require Dataloop items. Edit `tests/test_config.py`:

```python
TEST_ITEMS = {
    'pdf': {'item_id': 'your-pdf-item-id'},
    'doc': {'item_id': 'your-docx-item-id'},
}

TARGET_DATASET_ID = 'your-target-dataset-id'

PDF_CONFIG = {
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'chunking_strategy': 'recursive',
    'use_ocr': False,
}

DOC_CONFIG = {
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'chunking_strategy': 'recursive',
}
```

## Writing New Tests

### Testing Transforms

```python
from utils.extracted_data import ExtractedData
from utils.config import Config
import transforms

def test_my_transform():
    # Create test data
    data = ExtractedData(config=Config())
    data.content_text = "Test content"

    # Run transform
    result = transforms.my_transform(data)

    # Verify results
    assert isinstance(result, ExtractedData)
    assert result.current_stage == "expected_stage"
```

### Testing Extractors

```python
from utils.extracted_data import ExtractedData
from utils.config import Config
from apps.pdf_processor.pdf_extractor import PDFExtractor

def test_extractor():
    data = ExtractedData(config=Config())
    # Mock item if needed

    result = PDFExtractor.extract(data)

    assert result.current_stage == "extraction"
```

### Testing Config Validation

```python
from utils.config import Config
import pytest

def test_invalid_config():
    config = Config(max_chunk_size=-1)
    with pytest.raises(ValueError):
        config.validate()
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_config():
    return Config(max_chunk_size=100, chunk_overlap=10)

@pytest.fixture
def sample_data(sample_config):
    return ExtractedData(config=sample_config)
```
