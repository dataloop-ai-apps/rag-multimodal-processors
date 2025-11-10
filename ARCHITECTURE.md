# Architecture

## Overview

App-based architecture for processing PDF and DOC files:

```
Item → App (Extract → Process → Upload) → Chunks
```

Each file type has its own **App** class that orchestrates extraction, processing, and upload using shared utilities.

## Supported File Types

- **PDF** (.pdf) - `PDFApp`
- **DOC** (.docx) - `DOCApp`

## Components

### 1. Apps (`apps/`)

File-type specific processor classes. Each app:
- Imports and uses shared extractors and stages
- Orchestrates the processing pipeline
- Handles logging and error handling

**Structure:**
```
apps/
├── pdf-processor/
│   ├── pdf_app.py          # PDFApp class
│   ├── dataloop.json       # Dataloop app config
│   └── Dockerfile
└── doc-processor/
    ├── doc_app.py          # DOCApp class
    ├── dataloop.json
    └── Dockerfile
```

**Example App:**
```python
class PDFApp:
    def __init__(self, item: dl.Item, target_dataset: dl.Dataset, config: dict):
        self.item = item
        self.target_dataset = target_dataset
        self.config = config
        self.extractor = PDFExtractor()
        self.logger = logging.getLogger(f"PDFApp.{item.id[:8]}")

    def extract(self, data: dict) -> dict:
        extracted = self.extractor.extract(self.item, self.config)
        data.update(extracted.to_dict())
        return data

    def clean(self, data: dict) -> dict:
        data = stages.clean_text(data, self.config)
        data = stages.normalize_whitespace(data, self.config)
        return data

    def chunk(self, data: dict) -> dict:
        strategy = self.config.get('chunking_strategy', 'recursive')
        if strategy == 'recursive':
            data = stages.chunk_recursive(data, self.config)
        elif strategy == 'semantic':
            data = stages.llm_chunk_semantic(data, self.config)
        return data

    def upload(self, data: dict) -> dict:
        data = stages.upload_to_dataloop(data, self.config)
        return data

    def run(self) -> List[dl.Item]:
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = self.extract(data)
        data = self.apply_ocr(data)  # If enabled
        data = self.clean(data)
        data = self.chunk(data)
        data = self.upload(data)
        return data.get('uploaded_items', [])
```

### 2. Extractors (`extractors.py`)

Extract multimodal content from files:

```python
@dataclass
class ExtractedContent:
    text: str
    images: List[ImageContent]
    tables: List[TableContent]
    metadata: Dict[str, Any]
```

**Available Extractors:**
- `PDFExtractor` - Extract from PDF files
- `DocsExtractor` - Extract from DOCX files

### 3. Stages (`stages/`)

Shared processing functions with signature: `(data: dict, config: dict) -> dict`

- **preprocessing.py**: `clean_text`, `normalize_whitespace`, `remove_empty_lines`
- **chunking.py**: `chunk_recursive`, `chunk_by_sentence`, `chunk_by_paragraph`
- **ocr.py**: `ocr_enhance`, `describe_images_with_dataloop`
- **llm.py**: `llm_chunk_semantic`, `llm_summarize`
- **upload.py**: `upload_to_dataloop`, `upload_with_images`

All stages are imported and used by apps.

### 4. Main API (`main.py`)

Routes requests to appropriate apps based on MIME type:

```python
APP_REGISTRY = {
    'application/pdf': PDFApp,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCApp,
}

def process_item(item, target_dataset, config):
    app_class = APP_REGISTRY[item.mimetype]
    app = app_class(item, target_dataset, config)
    return app.run()
```

Convenience functions:
- `process_pdf(item, dataset, **config)`
- `process_doc(item, dataset, **config)`
- `process_batch(items, dataset, config)`

## Extension

### Add New File Type

To add support for a new file type (e.g., Excel):

**1. Create Extractor** (`extractors.py`):
```python
class XLSExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('application/vnd.ms-excel', 'XLS')

    def extract(self, item, config):
        # Extraction logic
        return ExtractedContent(text=extracted_text)

# Register
EXTRACTOR_REGISTRY['application/vnd.ms-excel'] = XLSExtractor
```

**2. Create App** (`apps/xls-processor/xls_app.py`):
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

**3. Register in Main API** (`main.py`):
```python
from xls_app import XLSApp

APP_REGISTRY['application/vnd.ms-excel'] = XLSApp
```

### Add New Stage

**1. Create Stage** (`stages/my_module.py`):
```python
def my_custom_stage(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom processing logic."""
    content = data.get('content', '')
    # Transform content
    data['content'] = transformed_content
    return data
```

**2. Export** (`stages/__init__.py`):
```python
from .my_module import my_custom_stage
__all__ = [..., 'my_custom_stage']
```

**3. Use in Apps**:
```python
def run(self):
    data = self.extract(data)
    data = stages.my_custom_stage(data, self.config)  # Use it!
    data = self.chunk(data)
    return data
```

## Configuration

Apps accept a configuration dictionary with these options:

```python
config = {
    # Chunking
    'max_chunk_size': 300,              # Maximum chunk size (100-10000)
    'chunk_overlap': 20,                # Overlap between chunks (0-500)
    'chunking_strategy': 'recursive',   # 'recursive', 'semantic', 'sentence', 'paragraph'

    # OCR (PDF only)
    'use_ocr': True,                    # Apply OCR to images
    'ocr_method': 'append_to_page',     # 'append_to_page', 'separate_chunks', 'combine_all'

    # LLM
    'llm_model_id': 'model-id',         # Required for semantic chunking

    # Logging
    'log_level': 'INFO',                # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
}
```

**Usage:**
```python
from main import process_pdf

chunks = process_pdf(
    item=pdf_item,
    target_dataset=dataset,
    use_ocr=True,
    max_chunk_size=500,
    chunking_strategy='semantic',
    llm_model_id='your-model-id',
    log_level='DEBUG'
)
```

All stages receive the same config dict for consistency.
