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

---

## Improvement Suggestions & Questions for Design/Product Team

This section documents design questions and technical decisions that need product/design team input.

### Configuration Architecture

#### Public API vs Internal Configuration
**Current State:** Config class has 23 fields, but only 7 are exposed in the public API (dataloop.json).

**Public API Fields (7):**
- `ocr_from_images` – enable OCR on images (default: False)
- `ocr_integration_method` – how OCR text is merged (append_to_page, separate_chunks, combine_all)
- `use_markdown_extraction` – extract Markdown formatting (default: False)
- `chunking_strategy` – text chunking method (recursive, fixed-size, nltk-sentence, nltk-paragraph, 1-chunk)
- `max_chunk_size` – max size per chunk (default: 300)
- `chunk_overlap` – overlap between chunks (default: 40)
- `to_correct_spelling` – enable spell correction (default: False)

**Internal-Only Fields (16):**
- Error handling: `error_mode`, `max_errors`
- Extraction: `extract_images`, `extract_tables`
- OCR internals: `ocr_method`, `ocr_model_id`
- Cleaning: `normalize_whitespace`, `remove_empty_lines`
- Upload: `remote_path`
- **Unused LLM:** `llm_model_id`, `generate_summary`, `extract_entities`, `translate`, `target_language`
- **Unused Vision:** `vision_model_id`

---

### Questions for Product/Design Team

#### 1. Unused Feature Fields (LLM & Vision)
**Status:** Fields exist but are not implemented

**LLM Fields:**
- `llm_model_id`, `generate_summary`, `extract_entities`, `translate`, `target_language`
- Functions exist but log "not yet implemented"
- Validation logic exists but can never be used

**Vision Fields:**
- `vision_model_id`
- `describe_images()` returns immediately without processing

**Questions:**
1. Should we remove these fields until there's active development?
2. Or keep them as "reserved for future use" with TODO comments?
3. If keeping, should we remove the validation logic that can never trigger?

**Impact:** Keeping them adds dead code and suggests features that don't exist

**Recommendation:** Remove entirely until implementation is planned

---

#### 2. Image & Table Extraction Control
**Status:** `extract_images=True` and `extract_tables=True` in Config but NOT in public API

**Current Behavior:**
- Images and tables are always extracted
- Users cannot disable this through the UI
- Config fields exist but are hardcoded to `True`

**Questions:**
1. Should these be:
   - **Option A:** Always enabled (remove from config, hardcode in extractors)
   - **Option B:** Added to public API as user-configurable checkboxes
   - **Option C:** Kept internal-only for future optimization use cases

2. Is there a use case for disabling image/table extraction in RAG pipelines?

**Recommendation:** Option A (always extract) - simplifies code and matches expected RAG behavior

---

#### 3. Config Class Structure
**Current Confusion:** Developers cannot easily tell which fields are user-facing vs internal

**Questions:**
1. Should we split into `PublicConfig` (7 fields) and `InternalConfig` (internal fields)?
2. Or document clearly which fields are public in code comments?

**Impact:**
- Splitting improves code clarity significantly
- Requires refactoring but no API changes

**Recommendation:** Split for clarity, or at minimum add clear documentation

---

### Field Naming & UX Questions

#### 4. Field Name: `to_correct_spelling`
**Current State:** Grammatically awkward field name

**Issues:**
- Grammar: "to correct" vs "correct"
- Misleading: Actually triggers "deep clean" (not just spell correction)
- UI Label: "Apply Text Cleaning" (clearer than field name)

**Questions:**
1. Can we rename to match behavior better?
   - Options: `apply_text_cleaning`, `deep_clean`, `enable_deep_clean`
2. Or keep field name as-is and rely on UI label?

**Impact:** Renaming is a breaking API change

**Recommendation:** Keep field name, ensure UI tooltip explains it's more than spelling

---

#### 5. OCR Integration Method Naming
**Current Options:**
- `append_to_page` - Appends OCR text after each page marker
- `separate_chunks` - Creates separate section at end with page markers
- `combine_all` - Combines all OCR at end without page markers

**Potential Confusion:**
- "separate_chunks" doesn't create separate chunks - it's a separate section
- Distinction between `separate_chunks` and `combine_all` is subtle

**Questions:**
1. Have users reported confusion about these names?
2. Should we rename for clarity (breaking change)?
3. Or improve tooltips/documentation?

**Recommendation:** Keep names unless user feedback indicates confusion

---

#### 6. Chunking Strategy & Field Dependencies
**Current Behavior:**
- `chunking_strategy='1-chunk'` ignores `max_chunk_size` and `chunk_overlap`
- `chunking_strategy='nltk-sentence'` ignores `chunk_overlap`
- All fields are always visible in UI

**Questions:**
1. Should we hide incompatible fields using `dependsOn` in dataloop.json?
2. Should we show warning messages when incompatible options are selected?
3. Or document the behavior and let users configure freely?

**Recommendation:** Use `dependsOn` to hide incompatible fields - reduces confusion

---

### Feature Roadmap Questions

#### 7. Batch OCR Implementation
**Current State:**
- `ocr_method` config field exists with options: `local`, `batch`, `auto`
- Only `local` (EasyOCR) works
- `batch` and `auto` fall back to `local` with warning log
- Field is internal-only (not exposed in UI)

**Questions:**
1. Is batch OCR planned for near-term development?
2. If not, should we remove this field from config?
3. Should users ever see this option?

**Recommendation:** Keep internal until batch OCR is implemented

---

### Documentation Gaps

#### 8. Pipeline Flow & Field Interactions
**Missing Documentation:**
1. Which config fields are public vs internal
2. Standard processing pipeline flow
3. How config fields affect pipeline behavior
4. Field interactions (e.g., `to_correct_spelling` → `deep_clean` instead of `clean`)

**Questions:**
1. Where should this be documented?
   - README.md for developers?
   - Inline tooltips for users?
   - Separate user guide?
2. Who is the audience - end users or developers?

**Recommendation:** Add to README.md and expand tooltips in dataloop.json

---

### Summary of Decisions Needed

| Priority | Question | Options | Recommendation |
|----------|----------|---------|----------------|
| **HIGH** | Remove unused LLM/Vision fields? | Remove / Keep as TODO | **Remove** |
| **HIGH** | Image/table extraction control? | Always-on / Configurable / Internal | **Always-on** |
| **MEDIUM** | Split Public/Internal Config? | Split / Document only | **Split** |
| **MEDIUM** | Rename `to_correct_spelling`? | Rename / Keep | **Keep** |
| **MEDIUM** | OCR method names confusing? | Rename / Keep / Improve tooltips | **Keep + tooltips** |
| **MEDIUM** | Hide incompatible chunking fields? | Yes / No | **Yes** |
| **LOW** | Batch OCR roadmap? | Plan / Remove | **Clarify** |
| **LOW** | Documentation location? | README / Tooltips / Guide | **Both** |

---

**Last Updated:** 2025-11-30
**Status:** Alpha - awaiting product team input on configuration architecture
