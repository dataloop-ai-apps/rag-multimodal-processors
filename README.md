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
    use_ocr=True,                    # Apply OCR to images
    ocr_method='append_to_page',     # How to integrate OCR text

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
| `ocr_method` | str | 'append_to_page' | How to integrate OCR: 'append_to_page', 'separate_chunks', 'combine_all' |
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

**1. Create Extractor** in `extractors.py`:
```python
class XLSExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('application/vnd.ms-excel', 'XLS')

    def extract(self, item, config):
        # Your extraction logic
        return ExtractedContent(text=extracted_text)

# Register the extractor
EXTRACTOR_REGISTRY['application/vnd.ms-excel'] = XLSExtractor
```

**2. Create App** in `apps/xls-processor/xls_app.py`:
```python
import logging
from typing import Dict, Any, List
import dtlpy as dl
from extractors import XLSExtractor
import stages

class XLSApp:
    def __init__(self, item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any] = None):
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}
        self.extractor = XLSExtractor()
        self.logger = logging.getLogger(f"XLSApp.{item.id[:8]}")

    def extract(self, data):
        extracted = self.extractor.extract(self.item, self.config)
        data.update(extracted.to_dict())
        return data

    def clean(self, data):
        data = stages.clean_text(data, self.config)
        return data

    def chunk(self, data):
        data = stages.chunk_recursive(data, self.config)
        return data

    def upload(self, data):
        data = stages.upload_to_dataloop(data, self.config)
        return data

    def run(self):
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = self.extract(data)
        data = self.clean(data)
        data = self.chunk(data)
        data = self.upload(data)
        return data.get('uploaded_items', [])
```

**3. Register App** in `main.py`:
```python
from xls_app import XLSApp

APP_REGISTRY['application/vnd.ms-excel'] = XLSApp
```

### Add a Custom Processing Stage

**1. Create Stage** in `stages/custom.py`:
```python
def my_custom_stage(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom processing logic."""
    content = data.get('content', '')
    # Transform content
    data['content'] = transformed_content
    return data
```

**2. Export** from `stages/__init__.py`:
```python
from .custom import my_custom_stage
__all__ = [..., 'my_custom_stage']
```

**3. Use in App**:
```python
# In your app's run() method
def run(self):
    data = self.extract(data)
    data = stages.my_custom_stage(data, self.config)  # Use your custom stage
    data = self.chunk(data)
    data = self.upload(data)
    return data.get('uploaded_items', [])
```

## Architecture

The system uses an **app-based architecture**:

```
apps/
├── pdf-processor/
│   └── pdf_app.py          # PDFApp - handles PDF processing
└── doc-processor/
    └── doc_app.py          # DOCApp - handles DOCX processing

extractors.py               # Extracts content from files (PDF, DOC only)
stages/                     # Shared processing utilities
main.py                     # Routes requests to appropriate app
```

Each app is self-contained and imports shared utilities (`extractors`, `stages`) as needed.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture details
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code
- **[tests/README.md](tests/README.md)** - Testing guide

## Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK](https://sdk-docs.dataloop.ai)
