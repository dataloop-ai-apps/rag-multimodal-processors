# Tests

Basic test suite for RAG Multimodal Processors.

## Running Tests

```bash
pytest tests/ -v
```

## Test Structure

- **test_static_methods.py** - Validates static method architecture
- **test_chunk_metadata.py** - Tests ChunkMetadata dataclass
- **test_data_types.py** - Tests data type classes
- **test_pdf_processor.py** - PDF processing integration tests
- **test_doc_processor.py** - DOCX processing integration tests

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
