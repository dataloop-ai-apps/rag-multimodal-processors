# Architecture

## Overview

**Type-safe, stateless architecture** using `ExtractedData` dataclass as the central data structure:

```
Item -> App (Extract -> Clean -> Chunk -> Upload) -> Chunks
```

Each file type is a self-contained processor that follows a consistent pipeline pattern with typed data flow.

## Supported File Types

- **PDF** (.pdf) - `PDFProcessor` with `PDFExtractor`
- **DOC** (.docx) - `DOCProcessor` with `DOCExtractor`

## Core Data Structure: ExtractedData

All pipeline operations use `ExtractedData` dataclass for type-safe data flow:

```python
@dataclass
class ExtractedData:
    # Input
    item: Optional[dl.Item] = None
    target_dataset: Optional[dl.Dataset] = None
    config: Config = field(default_factory=Config)

    # Extraction outputs
    content_text: str = ""
    images: List[ImageContent] = field(default_factory=list)
    tables: List[TableContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing outputs
    cleaned_text: str = ""
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[Dict[str, Any]] = field(default_factory=list)
    uploaded_items: List[Any] = field(default_factory=list)

    # Error tracking
    errors: ErrorTracker = field(default_factory=ErrorTracker)
    current_stage: str = "init"
```

## Components

### 1. Apps (`apps/`)

File-type specific processor classes. Each app:
- Uses `ExtractedData` throughout the pipeline
- Implements static methods for composable processing
- Calls shared transforms for reusable operations
- Has dedicated extractor module for file-specific logic

**Structure:**
```
apps/
├── __init__.py
├── pdf_processor/
│   ├── __init__.py
│   ├── app.py              # PDFProcessor class
│   ├── pdf_extractor.py    # PDFExtractor - extraction logic
│   ├── dataloop.json
│   └── Dockerfile
└── doc_processor/
    ├── __init__.py
    ├── app.py              # DOCProcessor class
    ├── doc_extractor.py    # DOCExtractor - extraction logic
    ├── dataloop.json
    └── Dockerfile
```

**Example Processor:**
```python
class PDFProcessor(dl.BaseServiceRunner):
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        return PDFExtractor.extract(data)

    @staticmethod
    def clean(data: ExtractedData) -> ExtractedData:
        return transforms.clean(data)

    @staticmethod
    def chunk(data: ExtractedData) -> ExtractedData:
        return transforms.chunk(data)

    @staticmethod
    def upload(data: ExtractedData) -> ExtractedData:
        return transforms.upload_to_dataloop(data)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, config: Optional[Dict] = None) -> List[dl.Item]:
        cfg = Config.from_dict(config or {})
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        data = PDFProcessor.extract(data)
        data = PDFProcessor.clean(data)
        data = PDFProcessor.chunk(data)
        data = PDFProcessor.upload(data)

        return data.uploaded_items
```

### 2. Extractors (`apps/*/extractor.py`)

Dedicated extraction modules with file-specific logic:

```python
class PDFExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        data.current_stage = "extraction"
        # File-specific extraction logic
        data.content_text = extracted_text
        data.images = extracted_images
        data.metadata['page_count'] = page_count
        return data
```

### 3. Transforms (`transforms/`)

**Pipeline operations with uniform signature:** `(data: ExtractedData) -> ExtractedData`

**Available Transforms:**

| Module | Functions |
|--------|-----------|
| `text_normalization.py` | `clean()`, `normalize_whitespace()`, `remove_empty_lines()`, `deep_clean()` |
| `chunking.py` | `chunk()`, `chunk_with_images()`, `TextChunker` |
| `ocr.py` | `ocr_enhance()`, `describe_images()` |
| `llm.py` | `llm_chunk_semantic()`, `llm_summarize()`, `llm_extract_entities()`, `llm_translate()` |
| `upload.py` | `upload_to_dataloop()`, `ChunkUploader` |

**Example Transform:**
```python
def clean(data: ExtractedData) -> ExtractedData:
    """Clean and normalize text content."""
    data.current_stage = "cleaning"
    content = data.content_text

    if not content:
        data.cleaned_text = ""
        return data

    # Apply cleaning
    content = content.strip()
    content = re.sub(r' +', ' ', content)
    content = re.sub(r'\n{3,}', '\n\n', content)

    data.cleaned_text = content
    data.metadata['cleaning_applied'] = True
    return data
```

### 4. Utils (`utils/`)

**Core utilities and data models:**

| Module | Purpose |
|--------|---------|
| `extracted_data.py` | `ExtractedData` dataclass - central pipeline structure |
| `config.py` | `Config` dataclass with validation |
| `errors.py` | `ErrorTracker` for error/warning tracking |
| `data_types.py` | `ImageContent`, `TableContent` data models |
| `chunk_metadata.py` | `ChunkMetadata` dataclass |
| `dataloop_helpers.py` | Dataloop integration helpers |

### 5. Main API (`main.py`)

Routes requests to appropriate apps based on MIME type:

```python
from apps import PDFProcessor, DOCProcessor

APP_REGISTRY: Dict[str, Type[dl.BaseServiceRunner]] = {
    'application/pdf': PDFProcessor,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCProcessor,
}

def process_item(item, target_dataset, config=None):
    app_class = APP_REGISTRY[item.mimetype]
    return app_class.run(item, target_dataset, config or {})
```

## Configuration

Configuration is handled through the `Config` dataclass with validation:

```python
@dataclass
class Config:
    # Error handling
    error_mode: Literal['stop', 'continue'] = 'continue'
    max_errors: int = 10

    # Extraction
    extraction_method: Literal['markdown', 'basic'] = 'markdown'
    extract_images: bool = True
    extract_tables: bool = True

    # OCR
    use_ocr: bool = False
    ocr_method: Literal['local', 'batch', 'auto'] = 'local'
    ocr_model_id: Optional[str] = None

    # Chunking
    chunking_strategy: Literal['recursive', 'fixed', 'sentence', 'none'] = 'recursive'
    max_chunk_size: int = 300
    chunk_overlap: int = 20

    # Cleaning
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    use_deep_clean: bool = False

    # LLM
    llm_model_id: Optional[str] = None
    generate_summary: bool = False
    extract_entities: bool = False
    translate: bool = False
    target_language: str = 'English'

    # Vision
    vision_model_id: Optional[str] = None

    # Upload
    remote_path: str = '/chunks'

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Validates chunk settings, OCR requirements, LLM requirements, etc.
```

## Error Handling

Error tracking through `ErrorTracker`:

```python
@dataclass
class ErrorTracker:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_errors: int = 10
    error_mode: str = 'continue'

    def add_error(self, message: str, stage: str = "") -> bool:
        """Add error and return whether to continue processing."""
        self.errors.append(f"[{stage}] {message}" if stage else message)

        if self.error_mode == 'stop':
            return False
        return len(self.errors) < self.max_errors
```

**Error Modes:**
- `'stop'`: Halt on first error
- `'continue'`: Allow up to `max_errors` before stopping

## Extension Guide

### Adding a New File Type

1. **Create Extractor** (`apps/xls_processor/xls_extractor.py`):
```python
from utils.extracted_data import ExtractedData

class XLSExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        data.current_stage = "extraction"
        # File-specific extraction
        data.content_text = extracted_text
        return data
```

2. **Create Processor** (`apps/xls_processor/app.py`):
```python
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

3. **Register** in `main.py`:
```python
APP_REGISTRY['application/vnd.ms-excel'] = XLSProcessor
```

### Adding a New Transform

```python
# transforms/custom.py
from utils.extracted_data import ExtractedData

def my_transform(data: ExtractedData) -> ExtractedData:
    """Custom transform with standard signature."""
    data.current_stage = "custom"
    content = data.get_text()
    # Transform logic
    data.cleaned_text = transformed_content
    return data
```

Export from `transforms/__init__.py`:
```python
from .custom import my_transform
```

## Testing

```bash
# Unit tests
pytest tests/test_core.py tests/test_extractors.py tests/test_transforms.py -v

# Integration tests (requires Dataloop auth)
pytest tests/test_processors.py -v
```

| Test File | Coverage |
|-----------|----------|
| `test_core.py` | Config, ErrorTracker, ExtractedData, ChunkMetadata |
| `test_extractors.py` | PDFExtractor, DOCExtractor |
| `test_transforms.py` | All transform functions |
| `test_processors.py` | Integration tests (PDF, DOC) |

## Design Principles

### Stateless Architecture
- All callable functions are static methods
- No instance variables - state passed through `ExtractedData`
- Thread-safe by design for concurrent processing

### Type Safety
- `ExtractedData` dataclass with typed fields
- `Config` dataclass with validation
- Transform signatures: `(ExtractedData) -> ExtractedData`

### Separation of Concerns
- **Extractors**: File-specific extraction logic
- **Transforms**: Reusable pipeline operations
- **Utils**: Core utilities and data models
- **Apps**: Compose extractors and transforms

### Concurrency Support
- Stateless functions enable parallel document processing
- No shared state or race conditions
- Each `ExtractedData` instance is independent
