# Architecture Review Plan

**Date:** 2025-11-25
**Status:** Alpha - Backwards compatibility NOT required
**Last Updated:** 2025-11-25

---

## Executive Summary

Review of `apps/`, `transforms/`, and `utils/` directories identified redundancies, type signature inconsistencies, and consolidation opportunities.

**Progress:** Priority 1 and Priority 2 completed. 129 tests passing.

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

### 2.3 Remaining Redundancies (Future Work)

| Issue | Status | Notes |
|-------|--------|-------|
| Duplicate text cleaning | Pending | `transforms/text_normalization.py` vs `utils/text_cleaning.py` |
| Duplicate whitespace normalization | Pending | Within `text_normalization.py` |
| Duplicate model execution | Pending | Multiple model calling patterns |
| Duplicate processor `__init__` | Keep | Each processor remains self-contained |

---

## 3. Type Signature & Stateless Design (Pending)

### 3.1 Design Principles

1. **Processors** (`apps/*/app.py`): Handle `ExtractedData` - orchestration layer
2. **Sub-modules** (chunkers, extractors, utils):
   - Use **explicit input/output types**
   - Must be **static/stateless** - no instance state, pure functions or static methods
   - All required data passed as parameters

### 3.2 Sub-modules to Refactor

| Module | Current State | Required Change |
|--------|---------------|-----------------|
| `transforms/chunking.py` - `TextChunker` | Instance with `__init__` | Convert to static methods |
| `apps/*/pdf_extractor.py` | Static but uses `ExtractedData` | Use explicit types |
| `apps/*/doc_extractor.py` | Static but uses `ExtractedData` | Use explicit types |
| `utils/ocr_utils.py` - `OCRProcessor` | Already static | Keep as-is |
| `utils/text_cleaning.py` | Module functions | Keep as-is (already stateless) |

### 3.3 Target Pattern - Extractors

**Current:**
```python
class PDFExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
```

**Target:**
```python
@dataclass
class PDFExtractionResult:
    text: str
    images: List[ImageContent]
    tables: List[TableContent]
    metadata: Dict[str, Any]
    errors: List[str]

class PDFExtractor:
    @staticmethod
    def extract(file_path: str, config: Config) -> PDFExtractionResult:
        """Explicit input/output types for testability."""
```

### 3.4 Target Pattern - TextChunker

**Current:**
```python
class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, strategy: str):
        self.chunk_size = chunk_size  # Instance state
```

**Target:**
```python
class TextChunker:
    @staticmethod
    def chunk(text: str, chunk_size: int, chunk_overlap: int, strategy: str) -> List[str]:
        """Pure static function - no state, explicit types."""
```

---

## 4. Config Alignment (Pending)

Options documented elsewhere but missing from `Config` class:

| Option | In Config? | Action |
|--------|------------|--------|
| `link_images_to_chunks` | No | Add or remove docs |
| `embed_images_in_chunks` | No | Add or remove docs |
| `image_marker_format` | No | Add or remove docs |
| `llm_model_id` | No | Add or remove docs |
| `vision_model_id` | No | Add or remove docs |
| `ocr_integration_method` | No | Add or remove docs |

---

## 5. Prioritized Action List

### Priority 1: Critical ✅ COMPLETED
- [x] Remove `main.py`
- [x] Add missing type hints to `utils/upload.py`

### Priority 2: High-Impact Consolidation ✅ COMPLETED
- [x] Remove `ExtractedContent` class from `utils/data_types.py`
- [x] Consolidate upload functions (removed `upload_chunks_bulk`)
- [x] Standardize all logger names to `"rag-preprocessor"`

### Priority 3: Architecture Improvements (Static/Stateless)
- [ ] Refactor `TextChunker` to use static methods (remove `__init__` state)
- [ ] Refactor extractors to use explicit types (return `ExtractionResult` dataclass)
- [ ] Consolidate model execution patterns to use `DataloopModelExecutor`
- [ ] Ensure all sub-modules are stateless pure functions

### Priority 4: Config Cleanup
- [ ] Align documented config options with `Config` class
- [ ] Remove unused imports (e.g., `deep_clean` in text_normalization.py)

### Priority 5: Optional Enhancements
- [ ] Integrate deep text cleaning as optional mode
- [ ] Implement all OCR integration methods

---

## 6. Current Architecture

```
apps/
    +-- pdf_processor/
    |       +-- app.py              # PDFProcessor (orchestration)
    |       +-- pdf_extractor.py    # PDFExtractor (needs explicit types)
    +-- doc_processor/
            +-- app.py              # DOCProcessor (orchestration)
            +-- doc_extractor.py    # DOCExtractor (needs explicit types)

transforms/                         # ExtractedData -> ExtractedData
    +-- text_normalization.py
    +-- chunking.py                 # TextChunker (needs static refactor)
    +-- ocr.py
    +-- llm.py

utils/
    +-- extracted_data.py           # ExtractedData dataclass
    +-- config.py                   # Config dataclass
    +-- errors.py                   # ErrorTracker
    +-- data_types.py               # ImageContent, TableContent
    +-- dataloop_helpers.py         # upload_chunks (single implementation)
    +-- dataloop_model_executor.py
    +-- ocr_utils.py                # OCRProcessor (already static)
    +-- text_cleaning.py            # Already stateless
    +-- upload.py                   # Transform wrappers only
```

---

## 7. Files Deleted

- [x] `main.py`
- [x] `ExtractedContent` class (from `utils/data_types.py`)
- [x] `upload_chunks_bulk()` function (from `utils/upload.py`)
- [x] `TestExtractedContent` tests (from `tests/test_data_types.py`)

---

## 8. Benefits of Static/Stateless Design

1. **Testability**: Static functions with explicit types are trivial to unit test
2. **Clarity**: Function signature shows all inputs and outputs
3. **No Hidden State**: No surprises from instance variables
4. **Parallelization**: Stateless functions can run in parallel safely
5. **Reusability**: Can call from anywhere without instantiation

---

## 9. Test Status

**129 tests passing** after Priority 1 and Priority 2 changes.

Test breakdown:
- `test_chunk_metadata.py` - 10 tests
- `test_data_types.py` - 4 tests (reduced from 8 after removing ExtractedContent)
- `test_extracted_data.py` - 24 tests
- `test_extractors.py` - 16 tests
- `test_transforms.py` - 32 tests
- `test_utils_config.py` - 16 tests
- `test_utils_errors.py` - 22 tests
- Other tests - 5 tests
