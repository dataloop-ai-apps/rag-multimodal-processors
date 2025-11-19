# Architecture

## Overview

App-based architecture for processing PDF and DOC files:

```
Item → App (Extract → Process → Upload) → Chunks
```

Each file type has its own **App** class that orchestrates extraction, processing, and upload using shared utilities.

## Supported File Types

- **PDF** (.pdf) - `PDFProcessor`
- **DOC** (.docx) - `DOCProcessor`

## Components

### 1. Apps (`apps/`)

File-type specific processor classes organized as a **proper Python package**. Each app:
- Imports and uses shared extractors and operations
- Orchestrates the processing pipeline
- Includes its own Dockerfile and Dataloop manifest
- Can be deployed independently to Dataloop platform

**Structure:**
```
apps/
├── __init__.py                  # Package exports PDFProcessor, DOCProcessor
├── pdf_processor/
│   ├── __init__.py             # Sub-package exports
│   ├── pdf_processor.py
│   ├── dataloop.json
│   ├── Dockerfile
│   └── README.md
└── doc_processor/
    ├── __init__.py             # Sub-package exports
    ├── doc_processor.py
    ├── dataloop.json
    ├── Dockerfile
    └── README.md
```

**Importing:**
```python
# Clean imports - no sys.path manipulation needed
from apps import PDFProcessor, DOCProcessor
```

**Example App:**
```python
class PDFProcessor:
    def __init__(self, item: dl.Item, target_dataset: dl.Dataset, config: dict):
        self.item = item
        self.target_dataset = target_dataset
        self.config = config
        self.extractor = PDFExtractor()
        self.logger = logging.getLogger(f"PDFProcessor.{item.id[:8]}")

    def extract(self, data: dict) -> dict:
        extracted = self.extractor.extract(self.item, self.config)
        data.update(extracted.to_dict())
        return data

    def clean(self, data: dict) -> dict:
        data = operations.clean_text(data, self.config)
        data = operations.normalize_whitespace(data, self.config)
        return data

    def chunk(self, data: dict) -> dict:
        strategy = self.config.get('chunking_strategy', 'recursive')
        if strategy == 'recursive':
            data = operations.chunk_recursive(data, self.config)
        elif strategy == 'semantic':
            data = operations.llm_chunk_semantic(data, self.config)
        return data

    def upload(self, data: dict) -> dict:
        data = operations.upload_to_dataloop(data, self.config)
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

### 2. Extractors (`extractors/`)

Extract multimodal content from files. Organized as a package:

```
extractors/
├── data_types.py        # ExtractedContent, ImageContent, TableContent data models
├── mixins.py            # DataloopModelMixin for model-based extractors
├── pdf_extractor.py     # PDFExtractor
├── docs_extractor.py    # DocsExtractor
├── ocr_extractor.py     # OCRExtractor
└── registry.py          # EXTRACTOR_REGISTRY and registration functions
```

**Available Extractors:**
- `PDFExtractor` - Extract from PDF files
- `DocsExtractor` - Extract from DOCX files
- `OCRExtractor` - Extract text from images using OCR

**Usage:**
```python
from extractors import PDFExtractor, ExtractedContent

extractor = PDFExtractor()
content = extractor.extract(item, config)
```

### 3. Operations (`operations/`)

**Pipeline interface layer** - Functions with standardized signature: `(data: dict, config: dict) -> dict`

Operations provide a uniform interface for composable pipelines. They:
- Extract parameters from the shared `data` dictionary
- Call utils implementation functions to do the actual work
- Put results back into the `data` dictionary
- Enable clean pipeline composition in app classes

**Available Operations:**
- **chunking.py**: `chunk_text`, `chunk_recursive_with_images`, `chunk_with_embedded_images`, `TextChunker`
- **preprocessing.py**: `clean_text`, `normalize_whitespace`, `remove_empty_lines`
- **text_cleaning.py**: `clean_text()` using unstructured.io library (used by preprocessing operations)
- **ocr.py**: `ocr_enhance`, `describe_images_with_dataloop`
- **llm.py**: `llm_chunk_semantic`, `llm_summarize`
- **upload.py**: `upload_to_dataloop`, `upload_with_images`

**Example Operation (Adapter Pattern):**
```python
# operations/upload.py
def upload_to_dataloop(data: Dict, config: Dict) -> Dict:
    """Pipeline interface - extracts from data dict, calls utils, returns updated dict"""
    from utils.dataloop_helpers import upload_chunks  # Import utils implementation

    # Extract from shared data dictionary
    chunks = data.get('chunks', [])
    item = data.get('item')
    target_dataset = data.get('target_dataset')

    # Call utils implementation
    uploaded_items = upload_chunks(chunks, item, target_dataset, '/chunks', {})

    # Put results back in data dictionary
    data['uploaded_items'] = uploaded_items
    return data  # Continue the pipeline
```

### 4. Utils (`utils/`)

**Implementation layer** - Infrastructure and reusable utilities with specific signatures.

Utils contains the actual implementation logic that operations call. Functions here:
- Have specific type signatures (not the generic `(data, config) -> data`)
- Can be used standalone outside of pipelines
- Contain the business logic and integrations

**Available Modules:**
- **dataloop_helpers.py**: `upload_chunks()`, `upload_images()`, `get_or_create_target_dataset()`
- **chunk_metadata.py**: `ChunkMetadata` class for standardized metadata
- **dataloop_model_executor.py**: `DataloopModelExecutor` base class for model execution

**Example Utils Function:**
```python
# utils/dataloop_helpers.py
def upload_chunks(
    chunks: List[str],           # Specific types, not generic dict
    original_item: dl.Item,
    target_dataset: dl.Dataset,
    remote_path: str,
    processor_metadata: Dict[str, Any]
) -> List[dl.Item]:              # Returns specific type
    """Direct implementation - can be used standalone"""
    # ... actual upload logic ...
    uploaded_items = target_dataset.items.upload(...)
    return uploaded_items
```

**Why Separate Operations and Utils?**

This is the [**Adapter Pattern**](https://en.wikipedia.org/wiki/Adapter_pattern):
- **Operations**: Uniform pipeline interface for composition
- **Utils**: Reusable implementations that can be used anywhere

Benefits:
1. ✅ **Flexibility**: Utils functions work standalone or in pipelines
2. ✅ **Testability**: Test utils logic separately from pipeline orchestration
3. ✅ **Composability**: Easy to build different pipelines by mixing operations
4. ✅ **Clear separation**: Pipeline interface vs. business logic

### 5. Main API (`main.py`)

Routes requests to appropriate apps based on MIME type using a simple registry pattern:

```python
# Clean imports - apps is now a proper package
from apps import PDFProcessor, DOCProcessor

APP_REGISTRY = {
    'application/pdf': PDFProcessor,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCProcessor,
}

def process_item(item, target_dataset, config):
    """Auto-detect file type and route to appropriate processor"""
    app_class = APP_REGISTRY[item.mimetype]
    app = app_class(item, target_dataset, config)
    return app.run()
```

**Convenience Functions:**
- `process_pdf(item, dataset, **config)` - Process PDF documents
- `process_doc(item, dataset, **config)` - Process Word documents
- `process_batch(items, dataset, config)` - Batch processing
- `get_supported_file_types()` - List supported MIME types

## Extension

### Add New File Type

To add support for a new file type (e.g., Excel):

**1. Create Extractor** (`extractors/xls_extractor.py`):
```python
from .data_types import ExtractedContent
import dtlpy as dl
from typing import Dict, Any

class XLSExtractor:
    def __init__(self):
        self.mime_type = 'application/vnd.ms-excel'
        self.name = 'XLS'

    def extract(self, item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        # Extraction logic
        return ExtractedContent(text=extracted_text)
```

Then register in `extractors/registry.py`:
```python
from .xls_extractor import XLSExtractor

EXTRACTOR_REGISTRY['application/vnd.ms-excel'] = XLSExtractor
```

And export from `extractors/__init__.py`:
```python
from .xls_extractor import XLSExtractor
__all__ = [..., 'XLSExtractor']
```

**2. Create App** (`apps/xls-processor/xls_app.py`):
```python
import logging
from typing import Dict, Any, List
import dtlpy as dl
from extractors import XLSExtractor
import operations

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
        data = operations.clean_text(data, self.config)
        return data

    def chunk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = operations.chunk_recursive(data, self.config)
        return data

    def upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = operations.upload_to_dataloop(data, self.config)
        return data

    def run(self) -> List[dl.Item]:
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = self.extract(data)
        data = self.clean(data)
        data = self.chunk(data)
        data = self.upload(data)
        return data.get('uploaded_items', [])
```

**3. Create Package Structure** (`apps/xls_processor/`):
```
apps/xls_processor/
├── __init__.py
├── xls_processor.py
├── dataloop.json
└── Dockerfile
```

**4. Export from Apps Package** (`apps/__init__.py`):
```python
from .xls_processor.xls_processor import XLSProcessor
__all__ = [..., 'XLSProcessor']
```

**5. Register in Main API** (`main.py`):
```python
from apps import PDFProcessor, DOCProcessor, XLSProcessor

APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

### Add New Operation

**Option A: Simple Operation (No Utils Implementation Needed)**

**1. Create Operation** (`operations/my_module.py`):
```python
def my_simple_operation(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline interface - simple transformation"""
    content = data.get('content', '')
    # Do transformation directly
    content = content.upper()  # Example
    data['content'] = content
    return data
```

**Option B: Operation with Utils Implementation (Recommended for Complex Logic)**

**1. Create Utils Implementation** (`utils/my_helper.py`):
```python
def transform_text(text: str, options: dict) -> str:
    """Utils implementation - reusable, testable"""
    # Complex logic here
    return transformed_text
```

**2. Create Operation Adapter** (`operations/my_module.py`):
```python
def my_custom_operation(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline interface - calls utils implementation"""
    from utils.my_helper import transform_text

    # Extract from data dict
    content = data.get('content', '')
    options = config.get('transform_options', {})

    # Call utils implementation
    transformed = transform_text(content, options)

    # Put back in data dict
    data['content'] = transformed
    return data
```

**3. Export Operation** (`operations/__init__.py`):
```python
from .my_module import my_custom_operation
__all__ = [..., 'my_custom_operation']
```

**4. Use in Apps**:
```python
def run(self):
    data = self.extract(data)
    data = operations.my_custom_operation(data, self.config)  # Use it!
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

All operations receive the same config dict for consistency.
