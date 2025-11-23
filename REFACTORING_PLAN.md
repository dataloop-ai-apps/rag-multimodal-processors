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
from typing import List, Dict, Any, Optional, Callable
import dtlpy as dl
from utils.data_types import ImageContent, TableContent

@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_index: int
    page_numbers: List[int] = field(default_factory=list)
    image_indices: List[int] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)  # Dataloop IDs after upload
    has_embedded_images: bool = False
    source_file: Optional[str] = None

@dataclass
class ExtractedData:
    """Central data structure that flows through the entire pipeline."""

    # === INPUT FIELDS ===
    item: Optional[dl.Item] = None
    target_dataset: Optional[dl.Dataset] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # === EXTRACTION OUTPUTS ===
    content_text: str = ""  # Extracted text content
    images: List[ImageContent] = field(default_factory=list)
    tables: List[TableContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === OCR OUTPUTS ===
    ocr_text: str = ""  # OCR extracted text
    ocr_integration_method: Optional[str] = None

    # === CLEANING OUTPUTS ===
    cleaned_content: str = ""  # Post-cleaning text
    cleaning_metadata: Dict[str, Any] = field(default_factory=dict)

    # === CHUNKING OUTPUTS ===
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[ChunkMetadata] = field(default_factory=list)
    chunking_strategy: Optional[str] = None

    # === UPLOAD OUTPUTS ===
    uploaded_items: List[dl.Item] = field(default_factory=list)
    uploaded_image_ids: Dict[str, str] = field(default_factory=dict)  # path -> item_id

    # === PROCESSING STATE ===
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_stage: str = "initialized"

    def add_error(self, error: str, stage: Optional[str] = None):
        """Add error with optional stage info."""
        if stage:
            self.errors.append(f"[{stage}] {error}")
        else:
            self.errors.append(error)

    def add_warning(self, warning: str):
        """Add warning message."""
        self.warnings.append(warning)

    def get_active_content(self) -> str:
        """Get the current active text content (cleaned if available, else raw)."""
        return self.cleaned_content if self.cleaned_content else self.content_text
```

---

## 2. SEPARATE EXTRACTION MODULES

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
        """Main entry point for processing."""
        config = config or {}

        # Initialize ExtractedData
        data = ExtractedData(
            item=item,
            target_dataset=target_dataset,
            config=config
        )

        # Execute pipeline steps directly
        data = PDFExtractor.extract(data)

        # Add OCR if configured
        if config.get('use_ocr', False):
            data = transforms.ocr_enhance(data)

        # Clean text
        data = transforms.clean_text(data)

        # Apply chunking strategy
        strategy = config.get('chunking_strategy', 'recursive')
        if strategy != 'none':
            if config.get('embed_images_in_chunks', False):
                data = transforms.chunk_with_embedded_images(data)
            else:
                data = transforms.chunk_text(data)

        # Upload to dataloop
        data = transforms.upload_to_dataloop(data)

        # Log any errors/warnings
        if data.errors:
            logger.error(f"Processing errors: {data.errors}")
        if data.warnings:
            logger.warning(f"Processing warnings: {data.warnings}")

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

### Phase 1: Core Data Structure
1. Create `utils/extracted_data.py` with `ExtractedData` class

### Phase 2: Extraction Modules
1. Create `apps/pdf_processor/pdf_extractor.py`
2. Create `apps/doc_processor/doc_extractor.py`
3. Move extraction logic from processors to extractors

### Phase 3: Transform Updates
1. Update each transform to use `ExtractedData`
2. Update transform signatures
3. Test each transform independently

### Phase 4: Processor Refactoring
1. Update `PDFProcessor` to directly call transforms
2. Update `DOCProcessor` to directly call transforms
3. Remove `process_document` method (unified with `run`)

### Phase 5: Main API Update
1. Update `main.py` with new interfaces
2. Update registry and helper functions

### Phase 6: Testing
1. Update all tests to use new data structures
2. Add tests for direct transform calls
3. Add tests for error handling

### Phase 7: Optional - TransformChain Enhancement (OPTIONAL)
1. Create `transforms/transform_chain.py` with `TransformChain` class
2. Enable composable pipeline configuration
3. Add support for dynamic pipeline building

---

## 7. KEY BENEFITS

### **Type Safety**
- All data fields are explicitly typed
- Functions have clear input/output types
- Errors caught at development time

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

### **Simplified Testing**
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

```python
config = {
    # Extraction
    'extraction_method': 'markdown',  # 'markdown' or 'basic'
    'extract_images': True,
    'extract_tables': True,

    # OCR
    'use_ocr': False,
    'ocr_integration_method': 'append',  # 'append', 'prepend', 'per_page', 'separate'

    # Chunking
    'chunking_strategy': 'recursive',  # 'recursive', 'sentence', 'paragraph', 'fixed-size', 'none'
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'embed_images_in_chunks': False,
    'link_images_to_chunks': True,

    # Upload
    'target_dataset_id': 'dataset_id',
    'upload_metadata': {},
}
```

---

## 9. EXAMPLE USAGE

```python
from main import process_pdf
from utils.extracted_data import ExtractedData

# Simple usage
chunks = process_pdf(
    item=pdf_item,
    target_dataset=dataset,
    use_ocr=True,
    max_chunk_size=500
)

# Advanced usage with custom transforms
from apps.pdf_processor.pdf_extractor import PDFExtractor

# Direct pipeline with custom transform
data = ExtractedData(item=pdf_item, target_dataset=dataset)
data = PDFExtractor.extract(data)
data = my_custom_transform(data)  # Custom transform
data = transforms.chunk_text(data)

# Access results
print(f"Extracted {len(data.images)} images")
print(f"Created {len(data.chunks)} chunks")
print(f"Errors: {data.errors}")

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

1. ✅ **Unified `ExtractedData` dataclass** with strong typing
2. ✅ **Separated extraction logic** into dedicated extractor modules
3. ✅ **Clear type flow** between all functions
4. ✅ **Single `run` method** entry point
5. ✅ **Direct transform calls** for simple, clear pipelines
6. ✅ **Stateless functions** for safe concurrent execution
7. ✅ **No legacy compatibility** - clean, modern architecture

The architecture provides a cleaner, more maintainable, type-safe, and concurrency-ready codebase suitable for production use.

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