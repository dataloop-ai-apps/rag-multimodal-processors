# RAG Document Processors

Modular, extensible processors for converting **PDF and DOC files** into RAG-ready chunks. Built on Dataloop with a type-safe pipeline architecture using `ExtractedData` dataclass.

## Supported File Types

- **PDF** (.pdf) - ML-enhanced text extraction with PyMuPDF Layout, optional OCR
- **Microsoft Word** (.docx) - Document processing with tables and images

## Key Features

- **Type-Safe Pipeline** - `ExtractedData` dataclass replaces dict-based data flow
- **Modular Architecture** - Clean separation between apps, transforms, and utilities
- **Easy File Type Addition** - Add new processors with consistent patterns
- **Pipeline Design** - Simple extract -> clean -> chunk -> upload flow
- **Static Methods** - Composable processing steps with no instance dependencies
- **Flexible OCR** - Dataloop models, EasyOCR fallback, or Tesseract
- **Multiple Chunking Strategies** - Recursive, semantic, sentence, paragraph
- **Error Handling** - Configurable error modes ('stop' or 'continue')
- **108 Unit Tests** - Comprehensive test coverage

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

# Process PDF with OCR
chunks = process_pdf(item, dataset, use_ocr=True, max_chunk_size=500)

# Process DOCX
chunks = process_doc(item, dataset, max_chunk_size=1000)
```

## Processing Options

### Basic Processing

```python
chunks = process_pdf(item, dataset)
```

Pipeline: Extract -> Clean -> Chunk -> Upload

### OCR for Scanned Documents

```python
chunks = process_pdf(item, dataset, use_ocr=True)
```

Pipeline: Extract -> OCR -> Clean -> Chunk -> Upload

### Custom Chunk Size

```python
chunks = process_pdf(
    item, dataset,
    max_chunk_size=500,
    chunk_overlap=50
)
```

### Chunking Strategies

```python
# Recursive (default) - Smart splitting on paragraphs, sentences, then characters
chunks = process_pdf(item, dataset, chunking_strategy='recursive')

# Sentence-based - Split on sentence boundaries
chunks = process_pdf(item, dataset, chunking_strategy='sentence')

# Semantic - Use LLM to identify semantic boundaries
chunks = process_pdf(
    item, dataset,
    chunking_strategy='semantic',
    llm_model_id='your-model-id'
)
```

### Batch Processing

```python
from main import process_batch

items = dataset.items.list()
results = process_batch(
    items=items,
    target_dataset=chunks_dataset,
    config={'use_ocr': True, 'max_chunk_size': 500}
)

for item_id, chunks in results.items():
    print(f"{item_id}: {len(chunks)} chunks")
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
    chunking_strategy='recursive',   # 'recursive', 'semantic', 'sentence', 'none'

    # OCR options (PDF only)
    use_ocr=True,                    # Apply OCR to images

    # Error handling
    error_mode='continue',           # 'stop' or 'continue' on errors
    max_errors=10,                   # Maximum errors before stopping

    # LLM options
    llm_model_id='your-model-id',    # Required for semantic chunking
)
```

### Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_chunk_size` | int | 300 | Maximum characters per chunk |
| `chunk_overlap` | int | 20 | Characters to overlap between chunks |
| `chunking_strategy` | str | 'recursive' | Strategy: 'recursive', 'semantic', 'sentence', 'none' |
| `use_ocr` | bool | False | Apply OCR to extract text from images |
| `error_mode` | str | 'continue' | Error handling: 'stop' or 'continue' |
| `max_errors` | int | 10 | Maximum errors before stopping |
| `llm_model_id` | str | None | Dataloop model ID for semantic chunking |

## Architecture

The system uses a **type-safe, stateless architecture** with `ExtractedData` as the central data structure:

```
Item -> App (Extract -> Clean -> Chunk -> Upload) -> Chunks
```

### Core Components

```
main.py                     # Entry point - routes to apps by MIME type

apps/                       # File-type processors
├── pdf_processor/
│   ├── app.py             # PDFProcessor class
│   └── pdf_extractor.py   # PDF extraction logic
└── doc_processor/
    ├── app.py             # DOCProcessor class
    └── doc_extractor.py   # DOCX extraction logic

transforms/                 # Pipeline transforms: (ExtractedData) -> ExtractedData
├── text_normalization.py  # clean(), normalize_whitespace(), remove_empty_lines()
├── chunking.py            # chunk(), chunk_with_images(), TextChunker
├── ocr.py                 # ocr_enhance(), describe_images()
└── llm.py                 # llm_chunk_semantic(), llm_summarize()

utils/                      # Core utilities and data models
├── extracted_data.py      # ExtractedData dataclass
├── config.py              # Config dataclass with validation
├── errors.py              # ErrorTracker for error handling
├── data_types.py          # ImageContent, TableContent
└── upload.py              # upload_to_dataloop()
```

### Key Design: ExtractedData

All transforms use `ExtractedData` dataclass for type-safe data flow:

```python
from utils.extracted_data import ExtractedData
from utils.config import Config

# Create typed data structure
data = ExtractedData(item=item, target_dataset=dataset, config=Config())

# Pipeline with type safety
data = PDFExtractor.extract(data)    # Populates content_text, images, tables
data = transforms.clean(data)         # Populates cleaned_text
data = transforms.chunk(data)         # Populates chunks, chunk_metadata
data = transforms.upload_to_dataloop(data)  # Populates uploaded_items

return data.uploaded_items
```

### Transform Signature

All transforms follow the signature: `(data: ExtractedData) -> ExtractedData`

```python
import transforms

# Text transforms
data = transforms.clean(data)
data = transforms.normalize_whitespace(data)

# Chunking transforms
data = transforms.chunk(data)
data = transforms.chunk_with_images(data)

# OCR transforms
data = transforms.ocr_enhance(data)

# Upload
data = transforms.upload_to_dataloop(data)
```

## Testing

108 unit tests covering all components:

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_transforms.py -v
pytest tests/test_extracted_data.py -v
pytest tests/test_extractors.py -v
```

Test breakdown:
- 16 config tests
- 20 error tracker tests
- 24 extracted data tests
- 16 extractor tests
- 32 transform tests

## Adding New File Types

### 1. Create Extractor

```python
# apps/xls_processor/xls_extractor.py
from utils.extracted_data import ExtractedData

class XLSExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        data.current_stage = "extraction"
        # Extract content from Excel file
        data.content_text = extracted_text
        data.metadata['processor'] = 'xls'
        return data
```

### 2. Create Processor

```python
# apps/xls_processor/app.py
import transforms
from utils.extracted_data import ExtractedData
from utils.config import Config

class XLSProcessor(dl.BaseServiceRunner):
    @staticmethod
    def run(item, target_dataset, config=None):
        cfg = Config.from_dict(config or {})
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        data = XLSExtractor.extract(data)
        data = transforms.clean(data)
        data = transforms.chunk(data)
        data = transforms.upload_to_dataloop(data)

        return data.uploaded_items
```

### 3. Register in main.py

```python
APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

## Adding New Transforms

```python
# transforms/custom.py
from utils.extracted_data import ExtractedData

def my_transform(data: ExtractedData) -> ExtractedData:
    """Custom transform following the standard signature."""
    data.current_stage = "custom"
    content = data.get_text()
    # Transform content
    data.cleaned_text = transformed_content
    return data
```

Export from `transforms/__init__.py`:
```python
from .custom import my_transform
```

Use in any processor:
```python
data = transforms.my_transform(data)
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture details
- **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - Refactoring implementation plan
- **[tests/README.md](tests/README.md)** - Testing guide

## Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK](https://sdk-docs.dataloop.ai)
