# Development Guide

Guide for Claude Code when working with this repository.

## Workflow

- Start in **PLAN mode** - gather info, create plans
- Move to **ACT mode** only when user types `ACT`
- Do NOT make changes without approval

## Architecture

System uses **app-based architecture** with simple function composition:

1. **Apps** (`apps/`) - File-type specific processor classes (PDFApp, DOCApp)
2. **Extractors** (`extractors.py`) - Extract content from files (PDF, DOC only)
3. **Stages** (`stages/`) - Shared processing functions with `(data: dict, config: dict) -> dict` signature
4. **Main API** (`main.py`) - Routes requests to appropriate apps

### Supported File Types
- **PDF** (.pdf) - via `apps/pdf-processor/pdf_app.py`
- **DOC** (.docx) - via `apps/doc-processor/doc_app.py`

## Repository Structure

```
rag-multimodal-processors/
├── main.py                      # Main API - routes to apps
├── extractors.py                # PDF & DOC extractors only
├── apps/                        # File type processors
│   ├── pdf-processor/
│   │   ├── pdf_app.py          # PDFApp class
│   │   ├── dataloop.json       # Dataloop app config
│   │   └── Dockerfile
│   └── doc-processor/
│       ├── doc_app.py          # DOCApp class
│       ├── dataloop.json
│       └── Dockerfile
├── stages/                      # Shared processing stages
│   ├── preprocessing.py        # Text cleaning
│   ├── chunking.py             # Chunking strategies
│   ├── ocr.py                  # OCR enhancement
│   ├── llm.py                  # LLM operations
│   └── upload.py               # Dataloop upload
├── chunkers/                    # Chunking implementations
├── extractors/                  # Shared utilities (OCR)
└── utils/                       # Helpers
```

## Key Files

- **Main API**: `main.py` - Entry point, routes to apps
- **PDF App**: `apps/pdf-processor/pdf_app.py`
- **DOC App**: `apps/doc-processor/doc_app.py`
- **Extractors**: `extractors.py` - PDF & DOC only
- **Stages**: `stages/` - Shared processing functions
- **Chunking**: `chunkers/text_chunker.py`
- **OCR**: `extractors/ocr_extractor.py`
- **Upload**: `utils/dataloop_helpers.py`

## Common Patterns

### Add New File Type

To add support for a new file type (e.g., Excel, PowerPoint):

1. **Create extractor** in `extractors.py`:
```python
class XLSExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('application/vnd.ms-excel', 'XLS')

    def extract(self, item, config):
        # Extract content from file
        return ExtractedContent(text="...")
```

2. **Register extractor**:
```python
EXTRACTOR_REGISTRY['application/vnd.ms-excel'] = XLSExtractor
```

3. **Create app directory**: `apps/xls-processor/`

4. **Create app class** in `apps/xls-processor/xls_app.py`:
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

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        extracted = self.extractor.extract(self.item, self.config)
        data.update(extracted.to_dict())
        return data

    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = stages.clean_text(data, self.config)
        return data

    def chunk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = stages.chunk_recursive(data, self.config)
        return data

    def upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = stages.upload_to_dataloop(data, self.config)
        return data

    def run(self) -> List[dl.Item]:
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = self.extract(data)
        data = self.clean(data)
        data = self.chunk(data)
        data = self.upload(data)
        return data.get('uploaded_items', [])
```

5. **Register in main.py**:
```python
from xls_app import XLSApp
APP_REGISTRY['application/vnd.ms-excel'] = XLSApp
```

### Add New Processing Stage

1. Create in `stages/`:
```python
def my_custom_stage(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process data in custom way."""
    content = data.get('content', '')
    # Transform content
    data['content'] = transformed_content
    return data
```

2. Export from `stages/__init__.py`:
```python
from .preprocessing import my_custom_stage
__all__ = [..., 'my_custom_stage']
```

3. Use in any app:
```python
def run(self):
    data = self.extract(data)
    data = stages.my_custom_stage(data, self.config)  # Use your custom stage
    data = self.chunk(data)
    return data
```

## Example Workflows

### Basic PDF Processing
```python
from main import process_pdf

chunks = process_pdf(
    item=pdf_item,
    target_dataset=chunks_dataset
)
```

### PDF with OCR
```python
chunks = process_pdf(
    item=pdf_item,
    target_dataset=chunks_dataset,
    use_ocr=True
)
```

### PDF with Custom Chunking
```python
chunks = process_pdf(
    item=pdf_item,
    target_dataset=chunks_dataset,
    max_chunk_size=500,
    chunk_overlap=50,
    chunking_strategy='semantic',
    llm_model_id='your-model-id'
)
```

### DOC Processing
```python
from main import process_doc

chunks = process_doc(
    item=docx_item,
    target_dataset=chunks_dataset,
    max_chunk_size=1000
)
```

### Batch Processing
```python
from main import process_batch

items = dataset.items.list()
results = process_batch(
    items=items,
    target_dataset=chunks_dataset,
    config={'use_ocr': True}
)
```

## Testing

```bash
python tests/test_pdf.py       # Edit test_config.py first
python tests/test_doc.py
```

See `tests/README.md` for details.

## Dataloop Integration

- All processing uses Dataloop items directly
- `item.download()` → process → `dataset.items.upload()`
- All LLM/vision via Dataloop models only

## Configuration

```python
config = {
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'chunking_strategy': 'recursive',
    'use_ocr': True,
    'llm_model_id': 'model-id',
}
```

All stages receive the same config dict.
