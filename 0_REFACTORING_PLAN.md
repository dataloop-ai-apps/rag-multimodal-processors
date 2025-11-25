# Refactoring Plan: Type-Safe Pipeline Architecture

## Overview

This plan transforms the current dict-based pipeline into a **type-safe, stateless architecture** with clear data flow, separated concerns, and composable processing pipelines. All callable functions are designed as static methods or pure functions to enable **safe concurrent execution**.

## Core Design Principles for Concurrency

### **Stateless Architecture**
- **ALL callable functions must be static methods or module-level functions**
- **NO instance variables** - all state passed through function parameters
- **NO global state mutations** - functions are pure and side-effect free
- **Thread-safe by design** - multiple workers can execute the same function concurrently

### **Why Static/Stateless?**
1. **Concurrency**: Enable parallel processing of multiple documents
2. **Scalability**: Functions can run in separate threads/processes without conflicts
3. **Testability**: Pure functions are easier to test in isolation
4. **Reliability**: No hidden state means predictable behavior

### **Implementation Rules**
```python
# ✅ GOOD - Static method (no instance state)
class PDFExtractor:
    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        # All state in parameters, no self
        return processed_data

# ✅ GOOD - Module-level function
def clean_text(data: ExtractedData) -> ExtractedData:
    # Pure function transformation
    return cleaned_data

# ❌ BAD - Instance method with state
class BadProcessor:
    def __init__(self):
        self.state = {}  # NO! Prevents concurrency

    def process(self, data):  # NO! Not static
        self.state['count'] += 1  # NO! Mutates instance state
```

---

## 1. NEW DATA STRUCTURE: `ExtractedData` Dataclass

Replace `Dict[str, Any]` with a comprehensive dataclass that exposes all fields:

### `utils/extracted_data.py` (NEW FILE)
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import dtlpy as dl
from utils.data_types import ImageContent, TableContent
from utils.config import Config  # Simple config
from utils.errors import ErrorTracker  # Simple error tracking

@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_index: int
    page_numbers: List[int] = field(default_factory=list)
    image_indices: List[int] = field(default_factory=list)
    source_file: Optional[str] = None

@dataclass
class ExtractedData:
    """Central data structure that flows through the entire pipeline."""

    # === INPUT FIELDS ===
    item: Optional[dl.Item] = None
    target_dataset: Optional[dl.Dataset] = None
    config: Config = field(default_factory=Config)  # Simple flat config

    # === EXTRACTION OUTPUTS ===
    content_text: str = ""  # Extracted text content
    images: List[ImageContent] = field(default_factory=list)
    tables: List[TableContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === PROCESSING OUTPUTS ===
    cleaned_content: str = ""  # Post-cleaning text
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[ChunkMetadata] = field(default_factory=list)
    uploaded_items: List[dl.Item] = field(default_factory=list)

    # === ERROR TRACKING (SIMPLE) ===
    errors: ErrorTracker = field(default_factory=ErrorTracker)
    processing_stage: str = "initialized"

    def __post_init__(self):
        """Setup error tracker with config."""
        self.errors.error_mode = self.config.error_mode
        self.errors.max_errors = self.config.max_errors

    def log_error(self, message: str) -> bool:
        """Log error and return whether to continue."""
        return self.errors.add_error(message, self.processing_stage)

    def log_warning(self, message: str):
        """Log warning (doesn't stop processing)."""
        self.errors.add_warning(message)

    def get_active_content(self) -> str:
        """Get the current active text content (cleaned if available, else raw)."""
        return self.cleaned_content if self.cleaned_content else self.content_text
```

---

## 2. CONFIGURATION AND ERROR HANDLING

### `utils/config.py` ✅ IMPLEMENTED
Flat configuration class with validation:
- **Config**: Single dataclass with all settings
- `validate()` method checks configuration consistency
- Two error modes: 'stop' or 'continue'
- `from_dict()` for creating from dictionaries
- `to_dict()` for serialization

### `utils/errors.py` ✅ IMPLEMENTED
Error tracking for pipeline processing:
- **ErrorTracker**: Tracks errors and warnings
- `add_error()` returns whether processing should continue
- Supports 'stop' mode (halt on first error) and 'continue' mode (allow up to max_errors)
- `get_summary()` for logging

---

## 3. SEPARATE EXTRACTION MODULES

Move extraction logic into dedicated modules:

### `apps/pdf_processor/pdf_extractor.py` (NEW FILE)
```python
"""PDF extraction logic separated from the main processor."""
import logging
from typing import Optional, List
from pathlib import Path
import pymupdf
import pymupdf4llm
import dtlpy as dl

from utils.data_types import ImageContent, TableContent
from utils.extracted_data import ExtractedData

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Handles all PDF extraction operations."""

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Main extraction entry point."""
        if not data.item:
            data.add_error("No item provided", "extraction")
            return data

        try:
            temp_path = data.item.download()

            # Choose extraction method based on config
            method = data.config.get('extraction_method', 'markdown')

            if method == 'markdown':
                text, images, tables, metadata = PDFExtractor._extract_with_markdown(
                    temp_path,
                    data.config
                )
            else:
                text, images, tables, metadata = PDFExtractor._extract_with_pymupdf(
                    temp_path,
                    data.config
                )

            # Populate ExtractedData
            data.content_text = text
            data.images = images
            data.tables = tables
            data.metadata.update(metadata)
            data.processing_stage = "extracted"

        except Exception as e:
            data.add_error(f"Extraction failed: {str(e)}", "extraction")
            logger.error(f"PDF extraction failed: {e}")

        return data

    @staticmethod
    def _extract_with_markdown(file_path: str, config: dict) -> tuple:
        """Extract using pymupdf4llm with ML layout detection."""
        # Implementation from current pdf_processor.py
        pass

    @staticmethod
    def _extract_with_pymupdf(file_path: str, config: dict) -> tuple:
        """Basic PyMuPDF extraction."""
        # Implementation from current pdf_processor.py
        pass

    @staticmethod
    def _extract_images_from_page(page, page_num: int) -> List[ImageContent]:
        """Extract images with bounding boxes from a page."""
        # Implementation from current pdf_processor.py
        pass
```

### `apps/doc_processor/doc_extractor.py` (NEW FILE)
```python
"""DOCX extraction logic separated from the main processor."""
import logging
from typing import List, Tuple
from docx import Document
import dtlpy as dl

from utils.data_types import ImageContent, TableContent
from utils.extracted_data import ExtractedData

logger = logging.getLogger(__name__)

class DOCExtractor:
    """Handles all DOCX extraction operations."""

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Main extraction entry point."""
        if not data.item:
            data.add_error("No item provided", "extraction")
            return data

        try:
            temp_path = data.item.download()
            doc = Document(temp_path)

            # Extract components
            text = DOCExtractor._extract_text(doc)
            images = DOCExtractor._extract_images(doc, temp_path)
            tables = DOCExtractor._extract_tables(doc)

            # Populate ExtractedData
            data.content_text = text
            data.images = images
            data.tables = tables
            data.metadata['page_count'] = len(doc.element.xpath('//w:sectPr'))
            data.processing_stage = "extracted"

        except Exception as e:
            data.add_error(f"Extraction failed: {str(e)}", "extraction")
            logger.error(f"DOCX extraction failed: {e}")

        return data

    @staticmethod
    def _extract_text(doc: Document) -> str:
        """Extract text from document paragraphs."""
        # Implementation from current doc_processor.py
        pass

    @staticmethod
    def _extract_images(doc: Document, file_path: str) -> List[ImageContent]:
        """Extract embedded images."""
        # Implementation from current doc_processor.py
        pass

    @staticmethod
    def _extract_tables(doc: Document) -> List[TableContent]:
        """Extract and convert tables to markdown."""
        # Implementation from current doc_processor.py
        pass
```

---

## 3. TYPED TRANSFORM FUNCTIONS

All transforms are pure functions with typed interfaces:

### Transform Function Signature
```python
from typing import Callable
from utils.extracted_data import ExtractedData

# All transforms follow this signature
Transform = Callable[[ExtractedData], ExtractedData]
```

### Updated transforms with types:

### `transforms/text_normalization.py` (UPDATED)
```python
"""Text normalization transforms with typed interfaces."""
from utils.extracted_data import ExtractedData

def clean_text(data: ExtractedData) -> ExtractedData:
    """Clean and normalize text content."""
    content = data.get_active_content()

    # Apply cleaning operations
    cleaned = normalize_whitespace(content)
    cleaned = remove_empty_lines(cleaned)

    data.cleaned_content = cleaned
    data.cleaning_metadata['operations'] = ['whitespace', 'empty_lines']
    data.processing_stage = "cleaned"

    return data

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Implementation
    pass

def remove_empty_lines(text: str) -> str:
    """Remove empty lines from text."""
    # Implementation
    pass
```

### `transforms/chunking.py` (UPDATED)
```python
"""Chunking transforms with typed interfaces."""
from utils.extracted_data import ExtractedData, ChunkMetadata

def chunk_text(data: ExtractedData) -> ExtractedData:
    """Chunk text based on configured strategy."""
    strategy = data.config.get('chunking_strategy', 'recursive')
    max_size = data.config.get('max_chunk_size', 300)
    overlap = data.config.get('chunk_overlap', 20)

    content = data.get_active_content()

    # Apply chunking
    chunks = []  # Actual chunking logic

    # Create metadata for each chunk
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        metadata = ChunkMetadata(
            chunk_index=i,
            page_numbers=[],  # Populate based on content
            source_file=data.item.name if data.item else None
        )
        chunk_metadata.append(metadata)

    data.chunks = chunks
    data.chunk_metadata = chunk_metadata
    data.chunking_strategy = strategy
    data.processing_stage = "chunked"

    return data
```

### `transforms/ocr.py` (UPDATED)
```python
"""OCR transforms with typed interfaces."""
from utils.extracted_data import ExtractedData

def ocr_enhance(data: ExtractedData) -> ExtractedData:
    """Enhance content with OCR from images."""
    if not data.images:
        return data

    # OCR logic
    ocr_text = ""  # Actual OCR implementation

    # Integrate OCR text based on method
    method = data.config.get('ocr_integration_method', 'append')

    if method == 'append':
        data.content_text = data.content_text + "\n\n" + ocr_text
    elif method == 'prepend':
        data.content_text = ocr_text + "\n\n" + data.content_text

    data.ocr_text = ocr_text
    data.ocr_integration_method = method
    data.processing_stage = "ocr_enhanced"

    return data
```

### `transforms/upload.py` (UPDATED)
```python
"""Upload transforms with typed interfaces."""
import dtlpy as dl
from utils.extracted_data import ExtractedData

def upload_to_dataloop(data: ExtractedData) -> ExtractedData:
    """Upload chunks to Dataloop."""
    if not data.chunks or not data.target_dataset:
        data.add_error("Missing chunks or target dataset", "upload")
        return data

    uploaded_items = []

    for chunk, metadata in zip(data.chunks, data.chunk_metadata):
        # Upload logic
        item = None  # Actual upload implementation
        if item:
            uploaded_items.append(item)

    data.uploaded_items = uploaded_items
    data.processing_stage = "uploaded"

    return data
```

---

## 4. REFACTORED PROCESSORS WITH CALLABLE PIPELINES

### `apps/pdf_processor/pdf_processor.py` (REFACTORED)
```python
"""Refactored PDF processor with direct transform calls."""
import logging
from typing import List, Dict, Any, Optional
import dtlpy as dl

from utils.extracted_data import ExtractedData
from .pdf_extractor import PDFExtractor
import transforms

logger = logging.getLogger(__name__)

class PDFProcessor(dl.BaseServiceRunner):
    """PDF document processor with configurable pipeline."""

    def __init__(self):
        """Initialize processor (stateless - no instance variables needed)."""
        pass

    @staticmethod
    def run(item: dl.Item,
            target_dataset: dl.Dataset,
            config: Optional[Dict[str, Any]] = None) -> List[dl.Item]:
        """Main entry point with simple error handling."""

        # Parse config (with fallback to defaults)
        try:
            cfg = Config.from_dict(config or {})
            cfg.validate()
        except ValueError as e:
            logger.error(f"Config error: {e}, using defaults")
            cfg = Config()

        # Initialize data
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        # Extract with simple error handling
        data.processing_stage = "extraction"
        try:
            data = PDFExtractor.extract(data)
        except Exception as e:
            if not data.log_error(f"Extraction failed: {e}"):
                return []  # Stop if error_mode is 'stop'

        # OCR (optional, skip on failure)
        if cfg.use_ocr:
            data.processing_stage = "ocr"
            try:
                data = transforms.ocr_enhance(data)
            except Exception as e:
                data.log_warning(f"OCR failed, continuing without: {e}")

        # Clean text
        data.processing_stage = "cleaning"
        data = transforms.clean_text(data)

        # Chunk with simple fallback
        data.processing_stage = "chunking"
        try:
            data = transforms.chunk_text(data)
        except Exception as e:
            # Try simpler chunking as fallback
            data.log_warning(f"Advanced chunking failed, trying fixed-size: {e}")
            data.config.chunking_strategy = 'fixed'
            data = transforms.chunk_text(data)

        # Upload
        data.processing_stage = "upload"
        data = transforms.upload_to_dataloop(data)

        # Log summary
        logger.info(f"Processing complete: {data.errors.get_summary()}")

        return data.uploaded_items
```

### `apps/doc_processor/doc_processor.py` (REFACTORED)
```python
"""Refactored DOC processor with direct transform calls."""
import logging
from typing import List, Dict, Any, Optional
import dtlpy as dl

from utils.extracted_data import ExtractedData
from .doc_extractor import DOCExtractor
import transforms

logger = logging.getLogger(__name__)

class DOCProcessor(dl.BaseServiceRunner):
    """DOCX document processor with configurable pipeline."""

    def __init__(self):
        """Initialize processor (stateless)."""
        pass

    @staticmethod
    def run(item: dl.Item,
            target_dataset: dl.Dataset,
            config: Optional[Dict[str, Any]] = None) -> List[dl.Item]:
        """Main entry point for processing."""
        config = config or {}

        # Initialize ExtractedData
        data = ExtractedData(
            item=item,
            target_dataset=target_dataset,
            config=config
        )

        # Execute pipeline steps directly
        data = DOCExtractor.extract(data)
        data = transforms.clean_text(data)

        # Chunking
        if config.get('chunking_strategy', 'recursive') != 'none':
            data = transforms.chunk_text(data)

        # Upload
        data = transforms.upload_to_dataloop(data)

        return data.uploaded_items
```

---

## 5. UPDATED MAIN API

### `main.py` (UPDATED)
```python
"""Main API with typed interfaces."""
from typing import List, Dict, Any, Optional, Type
import dtlpy as dl

from apps import PDFProcessor, DOCProcessor
from utils.extracted_data import ExtractedData

# Registry with proper typing
APP_REGISTRY: Dict[str, Type[dl.BaseServiceRunner]] = {
    'application/pdf': PDFProcessor,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCProcessor,
}

def process_item(item: dl.Item,
                 target_dataset: dl.Dataset,
                 config: Optional[Dict[str, Any]] = None) -> List[dl.Item]:
    """Process any supported document type."""
    mime_type = item.metadata.get('system', {}).get('mimetype', '')

    processor_class = APP_REGISTRY.get(mime_type)
    if not processor_class:
        raise ValueError(f"Unsupported file type: {mime_type}")

    return processor_class.run(item, target_dataset, config or {})

def process_pdf(item: dl.Item,
                target_dataset: dl.Dataset,
                **config) -> List[dl.Item]:
    """Process PDF with typed interface."""
    return PDFProcessor.run(item, target_dataset, config)

def process_doc(item: dl.Item,
                target_dataset: dl.Dataset,
                **config) -> List[dl.Item]:
    """Process DOCX with typed interface."""
    return DOCProcessor.run(item, target_dataset, config)

def process_batch(items: List[dl.Item],
                  target_dataset: dl.Dataset,
                  config: Optional[Dict[str, Any]] = None) -> List[List[dl.Item]]:
    """Process multiple items."""
    results = []
    for item in items:
        try:
            result = process_item(item, target_dataset, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {item.name}: {e}")
            results.append([])
    return results
```

---

## 6. IMPLEMENTATION PHASES

### Phase 0: Configuration and Error Handling ✅ COMPLETE
1. ✅ Created `utils/config.py` with `Config` class
2. ✅ Created `utils/errors.py` with `ErrorTracker`
3. ✅ Added validation in `Config.validate()` method
4. ✅ Added tests (36 passing)

### Phase 1: Core Data Structure ✅ COMPLETE
1. ✅ Created `utils/extracted_data.py` with `ExtractedData` class
2. ✅ Integrated `ErrorTracker` for error tracking
3. ✅ Added `Config` for typed configuration
4. ✅ Added tests (24 passing)

### Phase 2: Extraction Modules ✅ COMPLETE
1. ✅ Created `apps/pdf_processor/pdf_extractor.py`
2. ✅ Created `apps/doc_processor/doc_extractor.py`
3. ✅ Extractors work with `ExtractedData`
4. ✅ Added tests (16 passing)

### Phase 3: Transform Updates ✅ COMPLETE
1. ✅ Updated `transforms/text_normalization.py` - `clean()`, `normalize_whitespace()`, `remove_empty_lines()`
2. ✅ Updated `transforms/chunking.py` - `chunk()`, `chunk_with_images()`, `TextChunker`
3. ✅ Updated `transforms/ocr.py` - `ocr_enhance()`, `describe_images()`, `ocr_batch_enhance()`
4. ✅ Updated `transforms/llm.py` - `llm_chunk_semantic()`, `llm_summarize()`, etc.
5. ✅ Updated `utils/upload.py` - `upload_to_dataloop()`, `upload_metadata_only()`, `dry_run_upload()`
6. ✅ Updated `transforms/__init__.py` - exports new function names
7. ✅ All transforms now use signature: `(data: ExtractedData) -> ExtractedData`

### Phase 4: Processor Refactoring ✅ COMPLETE
1. ✅ Updated `PDFProcessor` to directly call transforms with ExtractedData
2. ✅ Updated `DOCProcessor` to directly call transforms with ExtractedData
3. ✅ Processors use simple transform calls: `transforms.clean(data)`, `transforms.chunk(data)`

### Phase 5: Main API Update ✅ COMPLETE
1. ✅ Updated `main.py` with proper type hints (`Optional`, `Type`)
2. ✅ Added `Config` import for type-safe configuration
3. ✅ Updated `APP_REGISTRY` with proper typing: `Dict[str, Type[dl.BaseServiceRunner]]`
4. ✅ Updated function signatures with `Optional[Dict[str, Any]]`

### Phase 6: Testing ✅ COMPLETE
1. ✅ Created `tests/test_transforms.py` with 32 tests covering:
   - Text cleaning transforms (`clean`, `normalize_whitespace`, `remove_empty_lines`)
   - Chunking transforms (`chunk`, `chunk_with_images`, `TextChunker`)
   - LLM transforms (`llm_chunk_semantic`, `llm_summarize`)
   - Transform signatures (all return `ExtractedData`)
   - Transform chaining (pipeline simulation)
2. ✅ All 108 tests passing:
   - 16 config tests
   - 20 error tracker tests
   - 24 extracted data tests
   - 16 extractor tests
   - 32 transform tests

### Phase 7: Optional - TransformChain Enhancement (OPTIONAL)
1. Create `transforms/transform_chain.py` with `TransformChain` class
2. Enable composable pipeline configuration
3. Add support for dynamic pipeline building

### Phase 8: Processor Simplification ✅ COMPLETE
Simplify processor classes by removing wrapper methods and calling transforms directly.

**Changes:**
1. ✅ Move chunking strategy logic into `transforms.chunk()` - centralize decision making
2. ✅ Remove static wrapper methods (`extract`, `clean`, `chunk`, `upload`)
3. ✅ Call transforms/extractors directly in static `run()` method

**Before:**
```python
class PDFProcessor:
    @staticmethod
    def extract(data): return PDFExtractor.extract(data)  # wrapper
    @staticmethod
    def clean(data): return transforms.clean(data)  # wrapper
    @staticmethod
    def chunk(data):  # duplicated logic in both processors
        if strategy == 'semantic': return transforms.llm_chunk_semantic(data)
        elif has_images: return transforms.chunk_with_images(data)
        else: return transforms.chunk(data)
```

**After:**
```python
class PDFProcessor:
    def __init__(self):
        # Just setup (NLTK, timeouts, etc.)

    @staticmethod
    def run(item, target_dataset, context):
        config = context.node.metadata.get('customNodeConfig', {})
        cfg = Config.from_dict(config)
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)
        data = PDFExtractor.extract(data)
        data = transforms.clean(data)
        data = transforms.chunk(data)  # smart chunk handles strategy
        data = transforms.upload_to_dataloop(data)
        return data.uploaded_items
```

**Smart `chunk()` in transforms/chunking.py:**
```python
def chunk(data: ExtractedData) -> ExtractedData:
    strategy = data.config.chunking_strategy
    if strategy == 'semantic':
        return llm_chunk_semantic(data)
    elif strategy == 'recursive' and data.has_images():
        return chunk_with_images(data)
    # ... existing chunking logic
```

---

## 7. KEY BENEFITS

### **Type Safety**
- All data fields are explicitly typed
- Configuration validated before processing
- Functions have clear input/output types
- IDE autocomplete and error detection

### **Error Management**
- Configurable error handling ('stop' or 'continue')
- Error and warning tracking per stage
- Maximum error threshold support
- Clear error summaries for debugging

### **Clear Data Flow**
- Visible pipeline with typed stages
- Each transform adds specific fields
- Processing stage tracking

### **Separation of Concerns**
- Extraction logic separated from processors
- Transforms are independent and composable
- Clean interface between components

### **Concurrency Support**
- All transforms are stateless functions
- Safe parallel processing of multiple documents
- No shared state or race conditions
- Thread-safe pipeline execution

### **Testing**
```python
def test_extraction():
    data = ExtractedData(item=test_item)
    result = PDFExtractor.extract(data)
    assert result.content_text != ""
    assert len(result.images) > 0

def test_pipeline():
    # Direct transform calls
    data = ExtractedData(item=test_item)
    data = PDFExtractor.extract(data)
    data = transforms.clean_text(data)
    data = transforms.chunk_text(data)
    assert len(data.chunks) > 0
```

### **Extensible Architecture**
```python
# Easy to add custom transforms
def my_custom_transform(data: ExtractedData) -> ExtractedData:
    # Custom processing logic
    data.metadata['custom_processed'] = True
    return data

# Use in pipeline
data = PDFExtractor.extract(data)
data = my_custom_transform(data)  # Insert custom step
data = transforms.chunk_text(data)
```

---

## 8. CONFIGURATION REFERENCE

### Typed Configuration with Validation
```python
from utils.config_schema import ProcessingConfig, ChunkingStrategy, ErrorHandlingMode

# Create typed configuration
config = ProcessingConfig(
    # Error handling
    error_mode=ErrorHandlingMode.CONTINUE,  # fail_fast, continue, strict
    max_errors=10,

    # Extraction configuration
    extraction=ExtractionConfig(
        method=ExtractionMethod.MARKDOWN,
        extract_images=True,
        extract_tables=True,
        image_quality=95,
        table_format='markdown'
    ),

    # OCR configuration
    ocr=OCRConfig(
        enabled=True,
        model_id='ocr-model-123',
        integration_method=OCRIntegrationMethod.APPEND,
        confidence_threshold=0.85,
        languages=['eng', 'fra']
    ),

    # Chunking configuration
    chunking=ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        max_chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=100,
        preserve_sentences=True,
        embed_images=False
    ),

    # Cleaning configuration
    cleaning=CleaningConfig(
        normalize_whitespace=True,
        remove_empty_lines=True,
        fix_encoding=True,
        min_line_length=1
    ),

    # Upload configuration
    upload=UploadConfig(
        batch_size=10,
        parallel_uploads=3,
        retry_attempts=3,
        timeout_seconds=300
    )
)

# Validate for specific file type
config.validate_for_file_type('application/pdf')

# Convert to/from dictionary
config_dict = config.to_dict()
config_from_dict = ProcessingConfig.from_dict(config_dict)
```

### Legacy Dictionary Format (auto-converted)
```python
# Still supported via from_dict()
config_dict = {
    'error_mode': 'continue',
    'max_errors': 10,
    'extraction': {
        'method': 'markdown',
        'extract_images': True
    },
    'ocr': {
        'enabled': True,
        'model_id': 'ocr-model-123'
    },
    'chunking': {
        'strategy': 'recursive',
        'max_chunk_size': 500
    }
}

# Auto-converted to typed config
config = ProcessingConfig.from_dict(config_dict)
```

---

## 9. EXAMPLE USAGE

### Basic Usage with Error Handling
```python
from main import process_pdf
from utils.config_schema import ProcessingConfig, ErrorHandlingMode

# Configure with error handling
config = ProcessingConfig(
    error_mode=ErrorHandlingMode.CONTINUE,
    max_errors=5,
    ocr={'enabled': True, 'model_id': 'ocr-123'},
    chunking={'strategy': 'recursive', 'max_chunk_size': 500}
)

# Process with error recovery
try:
    chunks = process_pdf(
        item=pdf_item,
        target_dataset=dataset,
        config=config.to_dict()
    )
    print(f"Successfully created {len(chunks)} chunks")
except Exception as e:
    print(f"Processing failed: {e}")
```

### Advanced Usage with Custom Error Recovery
```python
from utils.extracted_data import ExtractedData
from utils.error_handling import ErrorHandler, ErrorContext, ErrorCategory
from apps.pdf_processor.pdf_extractor import PDFExtractor

# Create data with typed config
config = ProcessingConfig(error_mode=ErrorHandlingMode.CONTINUE)
data = ExtractedData(item=pdf_item, target_dataset=dataset, config=config)

# Register custom recovery strategy
def custom_extraction_recovery(data, context):
    print("Attempting custom recovery...")
    # Implement fallback logic
    return data

data.error_context.recovery_strategies[ErrorCategory.EXTRACTION] = custom_extraction_recovery

# Execute with error handling
data = ErrorHandler.safe_execute(
    PDFExtractor.extract,
    data.error_context,
    ErrorCategory.EXTRACTION,
    'extraction',
    data
)

# Check processing state
if data.should_continue():
    data = transforms.chunk_text(data)
else:
    print(f"Stopping due to errors: {data.error_context.get_summary()}")

# Access detailed results
summary = data.get_processing_summary()
print(f"Processed with {summary['error_summary']['total_errors']} errors")
print(f"Created {len(data.chunks)} chunks")
print(f"Extracted {len(data.images)} images")

# Concurrent processing (enabled by stateless design)
from concurrent.futures import ThreadPoolExecutor

def process_single(item):
    return process_pdf(item, dataset, use_ocr=True)

# Process multiple PDFs in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    items = [pdf1, pdf2, pdf3, pdf4]
    results = list(executor.map(process_single, items))
```

---

## Summary

This refactoring delivers:

1. ✅ **Typed Configuration** with validation (`utils/config.py`)
2. ✅ **Error Tracking** with configurable behavior (`utils/errors.py`)
3. ✅ **Unified `ExtractedData` dataclass** with strong typing
4. ✅ **Separated extraction logic** into dedicated extractor modules
5. ✅ **Clear type flow** between all functions
6. ✅ **Single `run` method** entry point
7. ✅ **Direct transform calls** for clear pipelines
8. ✅ **Stateless functions** for concurrent execution

### Configuration Features:
- Single `Config` dataclass with all settings
- Validation via `validate()` method
- `from_dict()` / `to_dict()` for serialization

### Error Handling Features:
- Two modes: 'stop' (halt on first error) or 'continue' (allow up to max_errors)
- Error and warning tracking
- Stage-aware error messages

The architecture provides a maintainable, type-safe, and concurrency-ready codebase.

---

## 10. OPTIONAL ENHANCEMENT: TransformChain

For advanced use cases requiring dynamic pipeline composition, an optional `TransformChain` class can be added:

### `transforms/transform_chain.py` (OPTIONAL)
```python
"""Optional transform chain for composable pipelines."""
from typing import List, Callable
import logging
from utils.extracted_data import ExtractedData

logger = logging.getLogger(__name__)

# Type alias for transform functions
Transform = Callable[[ExtractedData], ExtractedData]

class TransformChain:
    """Composable chain of transforms for advanced pipeline configuration."""

    def __init__(self, *transforms: Transform):
        """Initialize with transform functions."""
        self.transforms = list(transforms)

    def __call__(self, data: ExtractedData) -> ExtractedData:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            try:
                data = transform(data)
                if data.errors:
                    logger.warning(f"Errors after {transform.__name__}: {data.errors}")
            except Exception as e:
                data.add_error(f"Transform {transform.__name__} failed: {str(e)}")
                logger.error(f"Transform failed: {e}", exc_info=True)
        return data

    def add(self, transform: Transform) -> 'TransformChain':
        """Add transform and return self for chaining."""
        self.transforms.append(transform)
        return self

    @staticmethod
    def from_config(config: dict) -> 'TransformChain':
        """Build chain from configuration dictionary."""
        chain = TransformChain()
        # Add transforms based on config
        return chain
```

### Usage Example with TransformChain
```python
from transforms.transform_chain import TransformChain

# Create reusable pipeline
pdf_pipeline = TransformChain(
    PDFExtractor.extract,
    transforms.ocr_enhance,
    transforms.clean_text,
    transforms.chunk_text,
    transforms.upload_to_dataloop
)

# Process multiple items with same pipeline
for item in items:
    data = ExtractedData(item=item, target_dataset=dataset)
    result = pdf_pipeline(data)
```

### Benefits of TransformChain
- **Dynamic composition**: Build pipelines at runtime based on configuration
- **Reusability**: Create once, use many times
- **Error handling**: Built-in error propagation and logging
- **Testing**: Easy to mock and test pipeline combinations

This remains **completely optional** - the core architecture works perfectly with direct transform calls.