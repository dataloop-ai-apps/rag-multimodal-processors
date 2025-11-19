# RAG Document Processors

Dataloop-based processors for converting **PDF and DOC files** into RAG-ready chunks. Each file type has its own app that uses shared utilities for extraction, processing, and upload.

## Supported File Types

- **PDF** (.pdf) - Text extraction with optional OCR
- **Microsoft Word** (.docx) - Document processing

## Quick Start

```python
import dtlpy as dl
from main import process_pdf, process_doc

# Get items
item = dl.items.get(item_id='your-item-id')
dataset = dl.datasets.get(dataset_id='your-dataset-id')

# Process PDF
chunks = process_pdf(item, dataset)
print(f"Created {len(chunks)} chunks")

# Process DOC with OCR
chunks = process_pdf(item, dataset, use_ocr=True, max_chunk_size=500)

# Process DOCX
chunks = process_doc(item, dataset, max_chunk_size=1000)
```

## Processing Options

### Basic Processing

Simple text extraction and chunking:

```python
chunks = process_pdf(item, dataset)
```

Pipeline: Extract → Clean → Chunk → Upload

### OCR for Scanned Documents

Extract text from images in PDFs:

```python
chunks = process_pdf(item, dataset, use_ocr=True)
```

Pipeline: Extract → OCR → Clean → Chunk → Upload

### Custom Chunk Size

Control how text is split:

```python
chunks = process_pdf(
    item, dataset,
    max_chunk_size=500,      # Larger chunks
    chunk_overlap=50         # More overlap between chunks
)
```

### Semantic Chunking

Use LLM to chunk by meaning instead of size:

```python
chunks = process_pdf(
    item, dataset,
    chunking_strategy='semantic',
    llm_model_id='your-model-id'
)
```

### Different Chunking Strategies

```python
# Recursive (default) - Smart splitting on paragraphs, sentences, then characters
chunks = process_pdf(item, dataset, chunking_strategy='recursive')

# Sentence-based - Split on sentence boundaries
chunks = process_pdf(item, dataset, chunking_strategy='sentence')

# Paragraph-based - Split on paragraph boundaries
chunks = process_pdf(item, dataset, chunking_strategy='paragraph')

# Semantic - Use LLM to identify semantic boundaries
chunks = process_pdf(
    item, dataset,
    chunking_strategy='semantic',
    llm_model_id='your-model-id'
)
```

### Batch Processing

Process multiple files at once:

```python
from main import process_batch

items = dataset.items.list()
results = process_batch(
    items=items,
    target_dataset=chunks_dataset,
    config={'use_ocr': True, 'max_chunk_size': 500}
)

# results is a dict: {item_id: [uploaded_chunks]}
for item_id, chunks in results.items():
    print(f"{item_id}: {len(chunks)} chunks")
```

### Debug Mode

Enable detailed logging:

```python
chunks = process_pdf(item, dataset, log_level='DEBUG')
```

## Configuration

All configuration options are passed as keyword arguments:

```python
chunks = process_pdf(
    item=pdf_item,
    target_dataset=dataset,

    # Chunking options
    max_chunk_size=300,              # Maximum chunk size (100-10000)
    chunk_overlap=20,                # Overlap between chunks (0-500)
    chunking_strategy='recursive',   # 'recursive', 'semantic', 'sentence', 'paragraph'

    # OCR options (PDF only)
    use_ocr=True,                        # Apply OCR to images
    ocr_integration_method='per_page',   # How to integrate OCR text

    # LLM options
    llm_model_id='your-model-id',    # Required for semantic chunking

    # Logging
    log_level='INFO'                 # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
)
```

### Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_chunk_size` | int | 300 | Maximum characters per chunk (100-10000) |
| `chunk_overlap` | int | 20 | Characters to overlap between chunks (0-500) |
| `chunking_strategy` | str | 'recursive' | Strategy: 'recursive', 'semantic', 'sentence', 'paragraph' |
| `use_ocr` | bool | False | Apply OCR to extract text from images (PDF only) |
| `ocr_integration_method` | str | 'per_page' | How to integrate OCR: 'per_page' (interleave by page), 'append' (at end), 'prepend' (at start), 'separate' (separate field) |
| `llm_model_id` | str | None | Dataloop model ID (required for semantic chunking) |
| `log_level` | str | 'INFO' | Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR' |

## Testing

Tests use real Dataloop items. Edit `tests/test_config.py` with your item IDs:

```python
# In tests/test_config.py
TEST_ITEMS = {
    'pdf': {'item_id': 'your-pdf-item-id'},
    'dataset': {'dataset_id': 'your-dataset-id'},
}
```

Then run:

```bash
python tests/test_pdf.py       # Test PDF processing
python tests/test_doc.py       # Test .docx processing
```

See [tests/README.md](tests/README.md) for details.

## Extension

### Add a New File Type

To add support for a new file type (e.g., Excel):

**1. Create Extractor** in `extractors/xls_extractor.py`:
```python
from .data_types import ExtractedContent
import dtlpy as dl
from typing import Dict, Any

class XLSExtractor:
    def __init__(self):
        self.mime_type = 'application/vnd.ms-excel'
        self.name = 'XLS'

    def extract(self, item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        # Your extraction logic
        return ExtractedContent(text=extracted_text)
```

Then register it in `extractors/registry.py`:
```python
from .xls_extractor import XLSExtractor

EXTRACTOR_REGISTRY['application/vnd.ms-excel'] = XLSExtractor
```

**2. Create App** in `apps/xls_processor/xls_processor.py`:
```python
import logging
from typing import Dict, Any, List
import dtlpy as dl
from extractors import XLSExtractor
import operations

class XLSProcessor(dl.BaseServiceRunner):
    def __init__(self, item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any] = None):
        super().__init__()
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}
        self.extractor = XLSExtractor()
        self.logger = logging.getLogger(f"XLSProcessor.{item.id[:8]}")

    def extract(self, data):
        extracted = self.extractor.extract(self.item, self.config)
        data.update(extracted.to_dict())
        return data

    def clean(self, data):
        data = operations.clean_text(data, self.config)
        return data

    def chunk(self, data):
        data = operations.chunk_text(data, self.config)
        return data

    def upload(self, data):
        data = operations.upload_to_dataloop(data, self.config)
        return data

    def run(self):
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = self.extract(data)
        data = self.clean(data)
        data = self.chunk(data)
        data = self.upload(data)
        return data.get('uploaded_items', [])
```

**3. Create Package Structure** `apps/xls_processor/__init__.py`:
```python
from .xls_processor import XLSProcessor
__all__ = ['XLSProcessor']
```

**4. Export from Apps Package** `apps/__init__.py`:
```python
from .xls_processor.xls_processor import XLSProcessor
__all__ = [..., 'XLSProcessor']
```

**5. Register App** in `main.py`:
```python
from apps import PDFProcessor, DOCProcessor, XLSProcessor

APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

### Add a Custom Processing Operation

**1. Create Operation** in `operations/custom.py`:
```python
def my_custom_operation(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom processing logic."""
    content = data.get('content', '')
    # Transform content
    data['content'] = transformed_content
    return data
```

**2. Export** from `operations/__init__.py`:
```python
from .custom import my_custom_operation
__all__ = [..., 'my_custom_operation']
```

**3. Use in App**:
```python
# In your app's run() method
def run(self):
    data = self.extract(data)
    data = operations.my_custom_operation(data, self.config)  # Use your custom operation
    data = self.chunk(data)
    data = self.upload(data)
    return data.get('uploaded_items', [])
```

## Architecture

The system uses an **app-based architecture** with clean separation of concerns:

```
main.py                     # Entry point - routes to apps via registry

apps/                       # File-type processors (proper Python package)
├── __init__.py            # Exports PDFProcessor, DOCProcessor
├── pdf_processor/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDFProcessor class
│   ├── dataloop.json
│   └── Dockerfile
└── doc_processor/
    ├── __init__.py
    ├── doc_processor.py   # DOCProcessor class
    ├── dataloop.json
    └── Dockerfile

extractors/                 # Content extraction package
├── __init__.py
├── data_types.py         # ExtractedContent, ImageContent, TableContent data models
├── mixins.py              # DataloopModelMixin for model-based extractors
├── pdf_extractor.py       # PDF extraction
├── docs_extractor.py      # DOCX extraction
├── ocr_extractor.py       # OCR extraction
└── registry.py           # Extractor registry

operations/                 # Pipeline interface layer
├── __init__.py            # Signature: (data: dict, config: dict) -> dict
├── chunking.py           # Chunking operations
├── preprocessing.py       # Text cleaning operations
├── text_cleaning.py      # Deep text cleaning (used by preprocessing)
├── ocr.py                # OCR operations
├── llm.py                # LLM operations
└── upload.py             # Upload operations

utils/                       # Implementation layer
├── __init__.py            # Reusable infrastructure utilities
├── dataloop_helpers.py   # Upload/dataset helpers
├── chunk_metadata.py     # Metadata models
└── dataloop_model_executor.py  # Model execution

tests/                      # All tests consolidated here
├── test_pdf.py
├── test_doc.py
├── test_integration.py
└── test_processors.py
```

### Key Design Pattern: Operations vs Utils

**Operations** (`operations/`) - Pipeline interface layer:
- Uniform signature: `(data: dict, config: dict) -> dict`
- Adapts utils functions to work in composable pipelines
- Manages the shared data dictionary that flows through pipelines

**Utils** (`utils/`) - Implementation layer:
- Specific type signatures for reusable functions
- Contains business logic and integrations
- Can be used standalone outside of pipelines

This **Adapter Pattern** provides flexibility, testability, and composability.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture details
- **[tests/README.md](tests/README.md)** - Testing guide

## Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK](https://sdk-docs.dataloop.ai)
