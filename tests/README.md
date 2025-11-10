# Tests

Tests for RAG Multimodal Processors using real Dataloop items.

## Setup

Edit `test_config.py` with your Dataloop item IDs:

```python
TEST_ITEMS = {
    'pdf': {'item_id': 'your-pdf-item-id'},
    'doc': {'item_id': 'your-docx-item-id'},
    'dataset': {'dataset_id': 'your-target-dataset-id'},
}
```

## Running Tests

```bash
python tests/test_pdf.py       # Test PDF processing
python tests/test_doc.py       # Test .docx processing
```

Each test file includes:
- **Extractor tests** - Verify file content extraction
- **Processing tests** - Verify full processing workflow
- **Verbose output** - Shows progress and results

## Test Files

### test_pdf.py
Tests PDF document processing:
- PDFExtractor initialization and extraction
- Full processing workflow (extract → clean → chunk → upload)
- Configuration validation

### test_doc.py
Tests .docx document processing:
- DocsExtractor initialization and extraction
- Full processing workflow
- Configuration validation

### test_config.py
Central configuration for all tests. Edit this file with your:
- Item IDs for different file types
- Target dataset ID
- Processing configuration

### conftest.py
Shared pytest fixtures and utilities.

## Test Output

Tests show detailed progress:

```
============================================================
Testing PDF Processing
============================================================

1️⃣ Testing extractor initialization...
   ✅ PDFExtractor initialized correctly

2️⃣ Testing content extraction...
   📥 Fetching item: 67890abc...
   ✅ Retrieved: document.pdf (application/pdf)
   📄 Extracting content...
   ✅ Extracted: 5000 chars, 3 images, 2 tables

3️⃣ Testing full processing workflow...
   ⚙️ Processing with 'basic' level...
   ✅ Success! Created 15 chunks
```

## Configuration

Each test uses configuration from `test_config.py`:

```python
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

## Requirements

- Python 3.8+
- Dataloop SDK (`dtlpy`)
- Valid Dataloop credentials
- Access to test items and datasets

## Troubleshooting

### Item Not Found
```
Error: Failed to retrieve item
```
**Solution**: Update `test_config.py` with valid item IDs

### Dataset Not Found
```
Error: Failed to retrieve dataset
```
**Solution**: Update `test_config.py` with valid dataset ID

### Authentication Error
```
Error: User is not logged in
```
**Solution**: Authenticate with Dataloop:
```bash
python -c "import dtlpy as dl; dl.setenv('prod')"
```
