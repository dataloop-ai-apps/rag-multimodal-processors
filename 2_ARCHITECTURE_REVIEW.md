# Architecture Review Plan

**Date:** 2025-11-25
**Status:** Alpha - Backwards compatibility NOT required
**Last Updated:** 2025-11-25

---

## Executive Summary

Review of `apps/`, `transforms/`, and `utils/` directories identified redundancies, type signature inconsistencies, and consolidation opportunities.

**Progress:** All priorities completed. 131 tests passing.

---

## 1. Critical Issues ✅ COMPLETED

### 1.1 Remove `main.py` ✅ DONE

- Deleted `main.py` from project root
- Convenience functions were not used by any production code
- Processors are invoked directly via Dataloop

### 1.2 Add Type Hints to `utils/upload.py` ✅ DONE

Added explicit `ExtractedData` type hints:
```python
def upload_to_dataloop(data: ExtractedData) -> ExtractedData:
def upload_metadata_only(data: ExtractedData) -> ExtractedData:
def dry_run_upload(data: ExtractedData) -> ExtractedData:
```

### 1.3 Remove Duplicate `ExtractedContent` Class ✅ DONE

- Removed `ExtractedContent` from `utils/data_types.py`
- Removed export from `utils/__init__.py`
- Removed `TestExtractedContent` tests from `tests/test_data_types.py`

---

## 2. Redundancies Consolidated ✅ COMPLETED

### 2.1 Upload Functions ✅ DONE

**Before:** Two implementations
- `utils/upload.py` - `upload_chunks_bulk()`
- `utils/dataloop_helpers.py` - `upload_chunks()`

**After:** Single implementation
- Removed `upload_chunks_bulk()` from `utils/upload.py`
- `upload_to_dataloop()` now uses `upload_chunks()` from `dataloop_helpers.py`
- Simplified imports in `utils/upload.py`

### 2.2 Logger Names ✅ DONE

**Before:** Mixed logger names
- Some files: `logging.getLogger(__name__)`
- Some files: `logging.getLogger("rag-preprocessor")`

**After:** All files use `logging.getLogger("rag-preprocessor")`

Files updated (11 total):
- `apps/pdf_processor/app.py`
- `apps/pdf_processor/pdf_extractor.py`
- `apps/doc_processor/app.py`
- `apps/doc_processor/doc_extractor.py`
- `transforms/chunking.py`
- `transforms/ocr.py`
- `utils/dataloop_helpers.py`
- `utils/dataloop_model_executor.py`
- `utils/ocr_utils.py`
- `utils/text_cleaning.py`
- `utils/errors.py`

### 2.3 Remaining Redundancies ✅ DONE

| Issue | Status | Notes |
|-------|--------|-------|
| Duplicate text cleaning | ✅ Done | Extracted `clean_text_basic()` static function |
| Duplicate whitespace normalization | ✅ Done | Extracted `normalize_whitespace_text()` static function |
| Duplicate model execution | Pending | Multiple model calling patterns exist |
| Duplicate processor `__init__` | Keep | Each processor remains self-contained |

---

## 3. Type Signature & Stateless Design ✅ COMPLETED

### 3.1 Design Principles

1. **Processors** (`apps/*/app.py`): Handle `ExtractedData` - orchestration layer
2. **Sub-modules** (chunkers, extractors, utils):
   - Use **explicit input/output types**
   - Must be **static/stateless** - no instance state, pure functions or static methods
   - All required data passed as parameters

### 3.2 Sub-modules Refactored ✅ DONE

| Module | Change Applied |
|--------|----------------|
| `transforms/chunking.py` - `TextChunker` | ✅ Converted to static methods |
| `apps/pdf_processor/pdf_extractor.py` | ✅ Simplified to directly populate `ExtractedData` |
| `apps/doc_processor/doc_extractor.py` | ✅ Simplified to directly populate `ExtractedData` |
| `transforms/text_normalization.py` | ✅ Added `TextNormalizer` class with static methods |
| `transforms/ocr.py` | ✅ Added `OCREnhancer` class (merged from `utils/ocr_utils.py`) |
| `transforms/llm.py` | ✅ Added `LLMProcessor` class with static methods |
| `transforms/upload.py` | ✅ Added `ChunkUploader` class (moved from `utils/upload.py`) |

### 3.3 Extractor Pattern (Simplified)

Extractors directly populate `ExtractedData` - no intermediate types needed:

```python
class PDFExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Extract content from PDF item."""
        data.current_stage = "extraction"
        # ... extraction logic directly populates data ...
        data.content_text = extracted_text
        data.images = images
        data.metadata = {...}
        return data
```

### 3.4 TextChunker Pattern (Implemented)

```python
class TextChunker:
    @staticmethod
    def chunk(
        text: str,
        chunk_size: int = 300,
        chunk_overlap: int = 20,
        strategy: str = 'recursive'
    ) -> List[str]:
        """Pure static function - no state, explicit types."""
```

---

## 4. Deep Text Cleaning Integration ✅ COMPLETED

### 4.1 New Transform: `deep_clean()`

Added `deep_clean()` transform to `transforms/text_normalization.py`:

```python
def deep_clean(data: ExtractedData) -> ExtractedData:
    """Apply aggressive text cleaning using unstructured.io library."""
```

**Features:**
- Extra whitespace removal
- Dash and bullet normalization
- Trailing punctuation removal
- Lowercase conversion
- Unicode quote replacement
- Non-ASCII character cleaning
- Broken paragraph grouping
- Ordered bullet cleaning

### 4.2 Config Option: `use_deep_clean`

Added to `utils/config.py`:
```python
use_deep_clean: bool = False  # Enable aggressive text cleaning
```

### 4.3 Usage

```python
import transforms

# Enable in config
config = Config(use_deep_clean=True)
data = ExtractedData(config=config)

# Use in pipeline
data = transforms.deep_clean(data)
```

---

## 5. OCR Integration Methods ✅ COMPLETED

### 5.1 New Config Option: `ocr_method`

Added to `utils/config.py`:
```python
ocr_method: Literal['local', 'batch', 'auto'] = 'local'
```

**Methods:**
| Method | Description | Requires Model ID |
|--------|-------------|-------------------|
| `local` | Use EasyOCR locally (default) | No |
| `batch` | Use Dataloop model for batch OCR | Yes |
| `auto` | Try batch first, fallback to local | Yes |

### 5.2 Updated Validation

- `ocr_model_id` only required when `ocr_method` is `'batch'` or `'auto'`
- Local OCR works without model ID

### 5.3 Usage

```python
# Local OCR (no model needed)
config = Config(use_ocr=True, ocr_method='local')

# Batch OCR via Dataloop
config = Config(use_ocr=True, ocr_method='batch', ocr_model_id='model-123')

# Auto: try batch, fallback to local
config = Config(use_ocr=True, ocr_method='auto', ocr_model_id='model-123')
```

### 5.4 Public OCR API

| Transform | Description |
|-----------|-------------|
| `ocr_enhance()` | Single entry point for OCR - routes based on `ocr_method` config |
| `describe_images()` | Generate image captions using vision model (different purpose) |

**Internal routing** (private functions):
- `_ocr_local()` - EasyOCR implementation
- `_ocr_batch()` - Dataloop batch processing
- `_ocr_auto()` - Try batch, fallback to local

---

## 6. Prioritized Action List

### Priority 1: Critical ✅ COMPLETED
- [x] Remove `main.py`
- [x] Add missing type hints to `utils/upload.py`

### Priority 2: High-Impact Consolidation ✅ COMPLETED
- [x] Remove `ExtractedContent` class from `utils/data_types.py`
- [x] Consolidate upload functions (removed `upload_chunks_bulk`)
- [x] Standardize all logger names to `"rag-preprocessor"`

### Priority 3: Architecture Improvements (Static/Stateless) ✅ COMPLETED
- [x] Refactor `TextChunker` to use static methods (remove `__init__` state)
- [x] Simplify extractors to directly populate `ExtractedData`
- [x] Add static classes: `TextNormalizer`, `OCREnhancer`, `LLMProcessor`, `ChunkUploader`
- [x] Consolidate OCR utils into `transforms/ocr.py`
- [x] Move upload transforms to `transforms/upload.py`
- [ ] Consolidate model execution patterns to use `DataloopModelExecutor` (optional)

### Priority 4: Optional Enhancements ✅ COMPLETED
- [x] Integrate deep text cleaning as optional mode (`deep_clean()` transform)
- [x] Implement OCR integration methods (`ocr_method` config option)

---

## 7. Current Architecture

```
apps/
    +-- pdf_processor/
    |       +-- app.py              # PDFProcessor (orchestration)
    |       +-- pdf_extractor.py    # PDFExtractor (static, explicit types)
    +-- doc_processor/
            +-- app.py              # DOCProcessor (orchestration)
            +-- doc_extractor.py    # DOCExtractor (static, explicit types)

transforms/                         # ExtractedData -> ExtractedData
    +-- text_normalization.py       # clean(), normalize_whitespace(), deep_clean()
    +-- chunking.py                 # chunk() + TextChunker (static methods)
    +-- ocr.py                      # ocr_enhance(), describe_images() (single OCR entry point)
    +-- llm.py

utils/
    +-- extracted_data.py           # ExtractedData dataclass
    +-- config.py                   # Config dataclass (with ocr_method, use_deep_clean)
    +-- errors.py                   # ErrorTracker
    +-- data_types.py               # ImageContent, TableContent
    +-- dataloop_helpers.py         # upload_chunks, cleanup helpers
    +-- dataloop_model_executor.py
    +-- chunk_metadata.py           # ChunkMetadata dataclass
```

---

## 8. Files Deleted

- [x] `main.py`
- [x] `ExtractedContent` class (from `utils/data_types.py`)
- [x] `upload_chunks_bulk()` function (from `utils/upload.py`)
- [x] `TestExtractedContent` tests (from `tests/test_data_types.py`)

---

## 9. Config Options Summary

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `error_mode` | `'stop'` \| `'continue'` | `'continue'` | Error handling mode |
| `max_errors` | `int` | `10` | Max errors before stopping |
| `extraction_method` | `'markdown'` \| `'basic'` | `'markdown'` | PDF extraction method |
| `extract_images` | `bool` | `True` | Extract images from documents |
| `extract_tables` | `bool` | `True` | Extract tables from documents |
| `use_ocr` | `bool` | `False` | Enable OCR processing |
| `ocr_method` | `'local'` \| `'batch'` \| `'auto'` | `'local'` | OCR method to use |
| `ocr_model_id` | `str` \| `None` | `None` | Dataloop model ID for batch OCR |
| `chunking_strategy` | `'recursive'` \| `'fixed'` \| `'sentence'` \| `'none'` | `'recursive'` | Text chunking strategy |
| `max_chunk_size` | `int` | `300` | Maximum chunk size |
| `chunk_overlap` | `int` | `20` | Overlap between chunks |
| `normalize_whitespace` | `bool` | `True` | Normalize whitespace |
| `remove_empty_lines` | `bool` | `True` | Remove empty lines |
| `use_deep_clean` | `bool` | `False` | Enable deep text cleaning |

---

## 10. Benefits of Static/Stateless Design

1. **Testability**: Static functions with explicit types are trivial to unit test
2. **Clarity**: Function signature shows all inputs and outputs
3. **No Hidden State**: No surprises from instance variables
4. **Parallelization**: Stateless functions can run in parallel safely
5. **Reusability**: Can call from anywhere without instantiation

---

## 11. Test Status

**131 tests passing** after all changes.

Test breakdown:
- `test_chunk_metadata.py` - 10 tests
- `test_data_types.py` - 4 tests
- `test_extracted_data.py` - 24 tests
- `test_extractors.py` - 16 tests
- `test_transforms.py` - 32 tests
- `test_utils_config.py` - 18 tests (added OCR method tests)
- `test_utils_errors.py` - 22 tests
- Other tests - 5 tests
