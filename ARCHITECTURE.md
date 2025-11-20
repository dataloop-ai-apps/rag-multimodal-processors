# Architecture

## Overview

**Modular app-based architecture** designed for easy extension with new file types:

```
Item → App (Extract → Clean → Chunk → Upload) → Chunks
```

Each file type is a self-contained processor that follows a simple, consistent pipeline pattern using shared transforms and utilities.

## Supported File Types

- **PDF** (.pdf) - `PDFProcessor`
- **DOC** (.docx) - `DOCProcessor`

## Components

### 1. Apps (`apps/`)

File-type specific processor classes organized as a **Python package**. Each app:
- Implements the same pipeline pattern: extract → clean → chunk → upload
- Uses static methods for composable processing steps
- Calls shared transforms for reusable operations
- Includes its own Dockerfile and Dataloop manifest for independent deployment

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

**Example App Pattern:**
```python
class PDFProcessor(dl.BaseServiceRunner):
    def __init__(self, item=None, target_dataset=None, config=None):
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}

    @staticmethod
    def extract(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from PDF file."""
        # File-specific extraction logic here
        return data

    @staticmethod
    def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        return transforms.clean_text(data, config)

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        return transforms.chunk_text(data, config)

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks to Dataloop."""
        return transforms.upload_to_dataloop(data, config)

    def run(self) -> List[dl.Item]:
        """Execute the pipeline."""
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = PDFProcessor.extract(data, self.config)
        data = PDFProcessor.clean(data, self.config)
        data = PDFProcessor.chunk(data, self.config)
        data = PDFProcessor.upload(data, self.config)
        return data.get('uploaded_items', [])
```

**Key Points:**
- Static methods enable simple composition without instance dependencies
- Each step receives and returns a shared `data` dictionary
- Transforms handle common operations (cleaning, chunking, upload)
- File-specific logic lives only in the `extract` method

### 2. Transforms (`transforms/`)

**Reusable pipeline operations** with uniform signature: `(data: dict, config: dict) -> dict`

Transforms provide composable operations that work across all file types:
- Extract parameters from the shared `data` dictionary
- Transform the data
- Put results back into the `data` dictionary
- Enable mixing and matching operations in any app

**Available Transforms:**
- **chunking.py**: Text chunking strategies (recursive, semantic, etc.)
- **preprocessing.py**: Text cleaning and normalization
- **text_cleaning.py**: Deep text cleaning using unstructured.io
- **ocr.py**: OCR enhancement for images
- **llm.py**: LLM-based operations (semantic chunking, summarization)
- **upload.py**: Chunk upload with metadata

**Example Transform:**
```python
# transforms/preprocessing.py
def clean_text(data: Dict, config: Dict) -> Dict:
    """Clean and normalize text content."""
    text = data.get('text', '')

    if config.get('to_correct_spelling', False):
        from utils.text_cleaning import clean_text as deep_clean
        text = deep_clean(text)

    data['text'] = text
    return data
```

**Why Uniform Signature?**
- **Composable**: Chain any transforms together
- **Reusable**: Same transform works in any app
- **Testable**: Easy to test in isolation
- **Flexible**: Add new transforms without changing apps

### 3. Utils (`utils/`)

**Shared utilities and data models** used by transforms and apps.

Utils provides infrastructure that multiple components depend on:
- Have specific type signatures (not generic `(data, config) -> data`)
- Can be used standalone outside of pipelines
- Contain implementation details and integrations

**Available Modules:**
- **dataloop_helpers.py**: Upload helpers and Dataloop integrations
- **chunk_metadata.py**: `ChunkMetadata` dataclass for standardized metadata
- **data_types.py**: `ExtractedContent`, `ImageContent`, `TableContent` data models
- **ocr_utils.py**: OCR utilities with multiple backends
- **dataloop_model_executor.py**: Model execution wrapper

**Example Utils:**
```python
# utils/chunk_metadata.py
@dataclass
class ChunkMetadata:
    """Standardized metadata for chunks."""
    source_item_id: str
    source_file: str
    chunk_index: int
    total_chunks: int

    def to_dict(self) -> dict:
        """Convert to dictionary for Dataloop upload."""
        return asdict(self)
```

### 4. Main API (`main.py`)

Routes requests to appropriate apps based on MIME type using a **registry pattern**:

```python
from apps import PDFProcessor, DOCProcessor

# Simple registry - just map MIME types to processor classes
APP_REGISTRY = {
    'application/pdf': PDFProcessor,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCProcessor,
}

def process_item(item, target_dataset, config):
    """Auto-detect file type and route to appropriate processor."""
    app_class = APP_REGISTRY[item.mimetype]
    app = app_class(item, target_dataset, config)
    return app.run()
```

**Adding a New File Type:**
```python
# 1. Import your new processor
from apps import PDFProcessor, DOCProcessor, XLSProcessor

# 2. Add to registry
APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor

# That's it! Now Excel files are supported.
```

## Extension Guide

### Adding a New File Type

The modular architecture makes adding new file types straightforward. Here's how to add Excel support:

**1. Create App** (`apps/xls_processor/xls_processor.py`):
```python
import logging
import tempfile
from typing import Dict, Any, List
import dtlpy as dl
import openpyxl
from utils.data_types import ExtractedContent
import transforms

class XLSProcessor(dl.BaseServiceRunner):
    def __init__(self, item=None, target_dataset=None, config=None):
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}

    @staticmethod
    def extract(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from Excel file - file-specific logic here."""
        item = data.get('item')
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)
            workbook = openpyxl.load_workbook(file_path)

            # Extract all text from all sheets
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
        """Clean text - reuse shared transform."""
        return transforms.clean_text(data, config)

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk text - reuse shared transform."""
        return transforms.chunk_text(data, config)

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks - reuse shared transform."""
        return transforms.upload_to_dataloop(data, config)

    def run(self) -> List[dl.Item]:
        """Execute pipeline - same pattern as other processors."""
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = XLSProcessor.extract(data, self.config)
        data = XLSProcessor.clean(data, self.config)
        data = XLSProcessor.chunk(data, self.config)
        data = XLSProcessor.upload(data, self.config)
        return data.get('uploaded_items', [])
```

Notice:
- Only `extract()` has file-specific logic
- `clean()`, `chunk()`, `upload()` just call shared transforms
- Same pipeline pattern as PDF and DOC processors

**2. Create Package** (`apps/xls_processor/__init__.py`):
```python
from .xls_processor import XLSProcessor
__all__ = ['XLSProcessor']
```

**3. Export** (`apps/__init__.py`):
```python
from .xls_processor.xls_processor import XLSProcessor
__all__ = [..., 'XLSProcessor']
```

**4. Register** (`main.py`):
```python
from apps import PDFProcessor, DOCProcessor, XLSProcessor
APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

**That's it!** Your new processor follows the same pattern and reuses all existing transforms.

### Adding a New Transform

**Option A: Simple Transform**

**1. Create Transform** (`transforms/my_module.py`):
```python
def uppercase_text(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Simple transform - do transformation directly."""
    text = data.get('text', '')
    data['text'] = text.upper()
    return data
```

**2. Export** (`transforms/__init__.py`):
```python
from .my_module import uppercase_text
__all__ = [..., 'uppercase_text']
```

**3. Use in Any App**:
```python
@staticmethod
def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    data = transforms.uppercase_text(data, config)  # Works in any processor!
    return data
```

**Option B: Transform with Utils (for complex logic)**

**1. Create Utils** (`utils/my_helper.py`):
```python
def complex_transformation(text: str, options: dict) -> str:
    """Reusable implementation."""
    # Complex logic here
    return transformed_text
```

**2. Create Transform** (`transforms/my_module.py`):
```python
def my_transform(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform calls utils implementation."""
    from utils.my_helper import complex_transformation

    text = data.get('text', '')
    options = config.get('my_options', {})
    transformed = complex_transformation(text, options)
    data['text'] = transformed
    return data
```

**Why This Pattern?**
- **Uniform interface**: All transforms have `(data, config) -> data` signature
- **Composable**: Chain transforms together easily
- **Reusable**: Same transform works in every processor
- **Simple**: Most transforms are just 5-10 lines of code

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

## Refactoring History

### Complete Architecture Refactoring

**Completed:** Full refactoring to modular app-based architecture with static methods and optimized uploads.

**Key Changes:**
1. **Metadata Standardization** - Implemented `ChunkMetadata` dataclass with validation at instantiation
2. **App Integration** - Merged extractors into self-contained app processors (removed `extractors/` directory)
3. **Static Methods** - Converted all processing methods to static for composable operations
4. **Bulk Upload** - Implemented pandas DataFrame bulk upload for efficient chunk uploads
5. **OCR Consolidation** - Unified OCR with conditional logic (Dataloop model + EasyOCR fallback)
6. **Directory Reorganization** - Created `transforms/` and `utils/` with clear separation of concerns
7. **Feature Completeness** - Added NLTK downloads, spell correction support, and batch OCR processing

**Architecture Pattern:**
- **Transforms** (`transforms/`) - Pipeline operations with uniform `(data: dict, config: dict) -> dict` signature
- **Utils** (`utils/`) - Reusable implementations with specific type signatures
- **Apps** (`apps/`) - Self-contained processors composing transforms and utils

This follows the **Adapter Pattern** - transforms adapt utils to a uniform pipeline interface.

**Benefits:**
- Static methods enable concurrent processing
- Self-contained apps with clear boundaries
- Standardized metadata across all processors
- Dataclass validation ensures data integrity
- Comprehensive test suite for all components
