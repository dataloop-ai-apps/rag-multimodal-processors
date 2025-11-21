# RAG Document Processors

Modular, extensible processors for converting **PDF and DOC files** into RAG-ready chunks. Built on Dataloop with a simple pipeline architecture that makes adding new file types straightforward.

## Supported File Types

- **PDF** (.pdf) - ML-enhanced text extraction with PyMuPDF Layout, optional OCR
- **Microsoft Word** (.docx) - Document processing with tables and images

## Key Features

🧩 **Modular Architecture** - Clean separation between apps, transforms, and utilities
📁 **Easy File Type Addition** - Add new processors in minutes with consistent patterns
🔄 **Pipeline Design** - Simple extract → clean → chunk → upload flow
🎯 **Static Methods** - Composable processing steps with no instance dependencies
🔍 **Flexible OCR** - Dataloop models, EasyOCR fallback, or Tesseract via pymupdf-layout
📊 **Multiple Chunking Strategies** - Recursive, semantic, sentence, paragraph
🏷️ **Rich Metadata** - Track page numbers, images, and extraction details per chunk

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

Basic test suite included. See [tests/README.md](tests/README.md) for configuration details.

## Adding New File Types

The modular architecture makes it straightforward to add support for new file types. Each processor follows the same simple pattern:

### Example: Adding Excel Support

**1. Create App** in `apps/xls_processor/xls_processor.py`:
```python
import logging
import tempfile
from typing import Dict, Any, List
import dtlpy as dl
import openpyxl  # Example: Excel library
from utils.data_types import ExtractedContent
import transforms

class XLSProcessor(dl.BaseServiceRunner):
    def __init__(self):
        pass  # No instance state needed

    @staticmethod
    def extract(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from Excel file."""
        item = data.get('item')
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)
            workbook = openpyxl.load_workbook(file_path)

            all_text = []
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = ' '.join([str(cell) for cell in row if cell])
                    all_text.append(row_text)

            result = ExtractedContent(
                text='\n'.join(all_text),
                metadata={'processor': 'xls', 'sheet_count': len(workbook.worksheets)}
            )
            data.update(result.to_dict())
        return data

    @staticmethod
    def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return transforms.clean_text(data, config)

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return transforms.chunk_text(data, config)

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return transforms.upload_to_dataloop(data, config)

    @staticmethod
    def process_document(item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Dataloop pipeline entry point."""
        config = context.node.metadata.get('customNodeConfig', {})
        return XLSProcessor.run(item, target_dataset, config)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any]) -> List[dl.Item]:
        """Main processing method."""
        data = {'item': item, 'target_dataset': target_dataset}
        data = XLSProcessor.extract(data, config)
        data = XLSProcessor.clean(data, config)
        data = XLSProcessor.chunk(data, config)
        data = XLSProcessor.upload(data, config)
        return data.get('uploaded_items', [])
```

**2. Create Package Structure** `apps/xls_processor/__init__.py`:
```python
from .xls_processor import XLSProcessor
__all__ = ['XLSProcessor']
```

**3. Export from Apps Package** `apps/__init__.py`:
```python
from .xls_processor.xls_processor import XLSProcessor
__all__ = [..., 'XLSProcessor']
```

**4. Register App** in `main.py`:
```python
from apps import PDFProcessor, DOCProcessor, XLSProcessor

APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

That's it! Your new processor is ready to use with the same pipeline pattern as existing processors.

### Add a Custom Transform

**1. Create Transform** in `transforms/custom.py`:
```python
def my_custom_transform(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom processing logic."""
    content = data.get('content', '')
    # Transform content
    processed = content.upper()  # Example transformation
    data['content'] = processed
    return data
```

**2. Export** from `transforms/__init__.py`:
```python
from .custom import my_custom_transform
__all__ = [..., 'my_custom_transform']
```

**3. Use in App**:
```python
# In your app's run() method
def run(self):
    data = {'item': self.item, 'target_dataset': self.target_dataset}
    data = self.extract(data)
    data = transforms.my_custom_transform(data, self.config)  # Use your custom transform
    data = self.chunk(data)
    data = self.upload(data)
    return data.get('uploaded_items', [])
```

The uniform `(data, config) -> data` signature makes transforms composable and interchangeable.

## Architecture

The system uses a **modular app-based architecture** where each file type is a self-contained processor that follows a simple pipeline pattern:

```
main.py                     # Entry point - routes to apps by MIME type

apps/                       # File-type processors (modular, independent)
├── __init__.py            # Exports PDFProcessor, DOCProcessor
├── pdf_processor/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDFProcessor with static pipeline methods
│   ├── dataloop.json
│   └── Dockerfile
└── doc_processor/
    ├── __init__.py
    ├── doc_processor.py   # DOCProcessor with static pipeline methods
    ├── dataloop.json
    └── Dockerfile

transforms/                 # Reusable pipeline transformations
├── __init__.py            # Uniform signature: (data, config) -> data
├── chunking.py           # Chunking strategies (recursive, semantic, etc.)
├── preprocessing.py       # Text cleaning and normalization
├── text_cleaning.py      # Deep text cleaning utilities
├── ocr.py                # OCR transformation operations
├── llm.py                # LLM-based transformations
└── upload.py             # Upload with per-chunk metadata

utils/                      # Shared utilities and data models
├── __init__.py
├── dataloop_helpers.py   # Upload helpers and Dataloop integrations
├── chunk_metadata.py     # ChunkMetadata dataclass with validation
├── data_types.py         # ExtractedContent, ImageContent, TableContent
├── ocr_utils.py          # OCR utilities with multiple backends
└── dataloop_model_executor.py  # Model execution wrapper
```

### Key Design Patterns

**Static Methods** - All pipeline steps are static for simple composition:
```python
# No instance needed - clean functional style
PDFProcessor.extract(data, config)
PDFProcessor.clean(data, config)
PDFProcessor.chunk(data, config)
PDFProcessor.upload(data, config)
```

**Transforms Layer** - Uniform interface makes operations composable:
- Signature: `(data: dict, config: dict) -> dict`
- Share data through a dictionary pipeline
- Mix and match transforms easily
- Add new transforms without changing apps

**Data Flow** - Simple dictionary-based pipeline:
```python
data = {'item': item, 'target_dataset': dataset}
data = extract(data, config)   # Adds 'text', 'images', 'metadata'
data = clean(data, config)     # Normalizes 'text'
data = chunk(data, config)     # Adds 'chunks'
data = upload(data, config)    # Adds 'uploaded_items'
return data['uploaded_items']
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture details
- **[tests/README.md](tests/README.md)** - Testing guide

## Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK](https://sdk-docs.dataloop.ai)
