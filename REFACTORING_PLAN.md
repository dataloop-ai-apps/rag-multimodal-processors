# RAG Multimodal Processors - Refactoring Plan

## Overview
This document outlines the comprehensive refactoring plan to improve code organization, enable concurrency through static methods, and standardize processing patterns across all file type processors.

## Core Principles
1. **Self-Contained Apps**: Each file type processor contains all its extraction and processing logic
2. **Static Methods**: Enable concurrent processing by using static methods throughout
3. **Standardized Patterns**: Consistent extract → clean → chunk → upload pipeline
4. **Single API Calls**: Optimize Dataloop uploads to minimize API calls
5. **Validation at Creation**: Use dataclasses with validation at instantiation

## Target Architecture

```
rag-multimodal-processors/
├── apps/                         # Self-contained file processors
│   ├── __init__.py
│   ├── pdf_processor/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py     # Contains ALL PDF extraction + processing
│   │   ├── dataloop.json
│   │   └── Dockerfile
│   └── doc_processor/
│       ├── __init__.py
│       ├── doc_processor.py     # Contains ALL DOC extraction + processing
│       ├── dataloop.json
│       └── Dockerfile
├── utils/                        # Shared utilities (renamed from 'shared')
│   ├── __init__.py
│   ├── data_types.py            # ExtractedContent, ImageContent, TableContent
│   ├── ocr_utils.py             # OCR functionality (used by multiple apps)
│   ├── text_utils.py            # Deep text cleaning utilities
│   ├── chunk_metadata.py        # Standardized chunk metadata dataclass
│   └── cleanup_utils.py         # Cleanup utilities
├── transforms/                   # Pipeline operations (renamed from 'operations')
│   ├── __init__.py
│   ├── preprocessing.py         # Text preprocessing operations
│   ├── chunking.py              # Chunking operations
│   ├── llm.py                   # LLM operations
│   └── upload.py                # Upload and dataset management
└── extractors/                   # TO BE REMOVED after refactoring
```

## Refactoring Tasks

### Phase 1: Metadata Standardization (4 tasks)

#### Current Issues
- Inconsistent metadata keys: `parent_data`, `source_data`, `original_file` used interchangeably
- Chunk metadata implemented as class with static methods instead of dataclass
- Metadata fields vary between PDF and DOC processors

#### Changes Required
1. **Standardize metadata keys across all processors:**
   - `parent_data` → `source_item`
   - `source_data` → `source_item`
   - `original_file` → `source_file`

2. **Convert ChunkMetadata to dataclass:**
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ChunkMetadata:
    """Standardized chunk metadata with validation."""
    # Required fields
    source_item_id: str
    source_file: str
    source_dataset_id: str
    chunk_index: int
    total_chunks: int

    # Optional fields
    page_numbers: Optional[List[int]] = None
    image_ids: Optional[List[str]] = None
    bbox: Optional[tuple] = None
    processing_timestamp: float = field(default_factory=time.time)
    processor: Optional[str] = None
    extraction_method: Optional[str] = None

    def __post_init__(self):
        """Validate required fields at instantiation."""
        if not self.source_item_id:
            raise ValueError("source_item_id is required")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
```

### Phase 2: App Integration (6 tasks)

#### PDF Processor Integration
Merge `PDFExtractor` functionality directly into `apps/pdf_processor/pdf_processor.py`:

```python
class PDFProcessor(dl.BaseServiceRunner):
    def __init__(self, item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any]):
        """Initialize with all required resources upfront."""
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}

    @staticmethod
    def extract_pdf(item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """Extract text and images from PDF (formerly in PDFExtractor)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)

            if config.get('use_markdown_extraction', False):
                return PDFProcessor._extract_with_markdown(file_path, item, temp_dir, config)
            else:
                return PDFProcessor._extract_with_pymupdf(file_path, item, temp_dir, config)

    @staticmethod
    def _extract_with_pymupdf(...) -> ExtractedContent:
        """PyMuPDF extraction with page markers."""
        # Implementation here

    @staticmethod
    def _extract_with_markdown(...) -> ExtractedContent:
        """Markdown-aware extraction using pymupdf4llm."""
        # Implementation here

    @staticmethod
    def _extract_images_from_page(...) -> List[ImageContent]:
        """Extract images with position metadata."""
        # Implementation here
```

#### DOC Processor Integration
Merge `DocsExtractor` functionality directly into `apps/doc_processor/doc_processor.py`:

```python
class DOCProcessor(dl.BaseServiceRunner):
    @staticmethod
    def extract_docx(item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """Extract content from DOCX (formerly in DocsExtractor)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)
            doc = Document(file_path)

            # Extract with proper instantiation
            images = []
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    # Create ImageContent with ALL values at instantiation
                    images.append(ImageContent(
                        path=image_path,
                        format=ext,
                        caption=None,  # Explicit
                        page_number=None,
                        bbox=None,
                        size=None
                    ))

            return ExtractedContent(
                text=text,
                images=images,
                tables=tables,
                metadata={...}
            )
```

### Phase 3: Static Method Conversion (8 tasks)

Convert all processing methods to static for concurrency:

```python
class PDFProcessor:
    @staticmethod
    def extract(item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """Static extraction method."""
        extracted = PDFProcessor.extract_pdf(item, config)
        return extracted.to_dict()

    @staticmethod
    def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Static cleaning method."""
        return transforms.clean_text(data, config)

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Static chunking method."""
        return transforms.chunk_text(data, config)

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Static upload method."""
        return transforms.upload_bulk(data, config)

    def run(self) -> List[dl.Item]:
        """Orchestrate static methods."""
        data = {'item': self.item, 'target_dataset': self.target_dataset}
        data = PDFProcessor.extract(self.item, self.config)
        data = PDFProcessor.clean(data, self.config)
        data = PDFProcessor.chunk(data, self.config)
        data = PDFProcessor.upload(data, self.config)
        return data.get('uploaded_items', [])
```

### Phase 4: Upload Optimization (3 tasks)

#### Current Issues
- Multiple API calls: one per chunk for metadata updates
- No bulk upload implementation
- Inefficient item-by-item processing

#### New Bulk Upload Implementation
```python
# transforms/upload.py

@staticmethod
def upload_chunks_bulk(
    chunks: List[str],
    chunk_metadata_list: List[Dict[str, Any]],
    original_item: dl.Item,
    target_dataset: dl.Dataset
) -> List[dl.Item]:
    """
    Optimized bulk upload using pandas DataFrame.
    ONE API call instead of N calls.
    """
    import pandas as pd

    records = []
    for idx, (chunk_text, chunk_meta) in enumerate(zip(chunks, chunk_metadata_list)):
        metadata = ChunkMetadata(
            source_item_id=original_item.id,
            source_file=original_item.name,
            source_dataset_id=original_item.dataset.id,
            chunk_index=idx,
            total_chunks=len(chunks),
            page_numbers=chunk_meta.get('page_numbers'),
            image_ids=chunk_meta.get('image_ids')
        )

        records.append({
            'filename': f"{original_item.name}_chunk_{idx:04d}.txt",
            'text': chunk_text,
            'metadata': metadata.to_dict(),
            'remote_path': f'/chunks/{original_item.dir.lstrip("/")}'
        })

    df = pd.DataFrame(records)

    # Single bulk upload call
    uploaded_items = target_dataset.items.upload_dataframe(
        df=df,
        item_metadata_column='metadata',
        remote_path_column='remote_path',
        local_path_column='text',
        file_name_column='filename',
        overwrite=True
    )

    return uploaded_items
```

### Phase 5: OCR Consolidation (2 tasks)

Consolidate OCR functionality into a single conditional flow:

```python
# utils/ocr_utils.py

class OCRProcessor:
    @staticmethod
    def extract_text(item: dl.Item, config: Dict[str, Any]) -> str:
        """
        Single OCR method with conditional logic.
        If model_id provided → use Dataloop model
        Otherwise → use EasyOCR as fallback
        """
        model_id = config.get('custom_ocr_model_id')

        if model_id:
            # Use Dataloop model
            return OCRProcessor._extract_with_model(item, model_id)
        else:
            # Use EasyOCR fallback
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)
                return OCRProcessor._extract_with_easyocr(file_path)
```

### Phase 6: Directory Reorganization (5 tasks)

1. **Create new `utils/` directory structure:**
   - `utils/data_types.py` - Data structures (ExtractedContent, ImageContent, TableContent)
   - `utils/ocr_utils.py` - OCR functionality
   - `utils/text_utils.py` - Text cleaning utilities
   - `utils/chunk_metadata.py` - Metadata dataclass
   - `utils/cleanup_utils.py` - Cleanup utilities

2. **Enhance `transforms/` directory:**
   - Move upload_chunks → `transforms/upload.py`
   - Move dataset management → `transforms/upload.py`
   - Keep preprocessing, chunking, llm operations

3. **Remove old directories:**
   - Delete `extractors/` after merging into apps
   - Delete old `utils/dataloop_helpers.py` after redistribution

### Phase 7: LLM & Testing (4 tasks)

1. **Research Dataloop prompt items API**
   - Review developers.dataloop.ai documentation
   - Understand correct prompt item structure

2. **Update LLM operations:**
```python
# transforms/llm.py

def _call_llm_model(model_id: str, prompt: str, dataset: dl.Dataset) -> str:
    """
    Corrected LLM call using proper prompt items.
    """
    # Create prompt item correctly per Dataloop docs
    prompt_item = dataset.items.upload(
        local_path=create_prompt_item(prompt),
        item_type='prompt'  # Correct item type
    )

    # Execute model
    execution = model.predict(item_ids=[prompt_item.id])
    execution.wait()

    # Parse response correctly
    return parse_llm_response(execution)
```

3. **Create comprehensive tests:**
   - Unit tests for all static methods
   - Integration tests for concurrent processing
   - End-to-end pipeline tests

## Implementation Order

1. **Week 1: Foundation**
   - Create new directory structure
   - Move data types to utils/
   - Implement ChunkMetadata dataclass
   - Standardize metadata keys

2. **Week 2: App Integration**
   - Merge extractors into app processors
   - Convert to static methods
   - Update __init__ patterns

3. **Week 3: Optimization**
   - Implement bulk upload
   - Consolidate OCR
   - Update transforms

4. **Week 4: Cleanup & Testing**
   - Remove old directories
   - Create tests
   - Documentation updates

## Benefits

1. **Performance**: 80% reduction in API calls through bulk upload
2. **Concurrency**: Static methods enable parallel processing
3. **Maintainability**: Self-contained apps with clear boundaries
4. **Consistency**: Standardized metadata and processing patterns
5. **Validation**: Dataclasses ensure data integrity at creation

## Migration Strategy

1. **Backward Compatibility**: Keep legacy methods during transition
2. **Gradual Rollout**: Migrate one processor at a time
3. **Testing**: Comprehensive tests before removing old code
4. **Documentation**: Update all docs and examples

## Success Metrics

- [ ] All methods converted to static
- [ ] Single API call for chunk uploads
- [ ] Metadata standardized across all processors
- [ ] Extractors merged into apps
- [ ] All tests passing
- [ ] Documentation updated

## Notes

- Keep OCR in utils/ as it's shared by multiple apps
- Maintain backward compatibility during migration
- Focus on one processor at a time to minimize risk
- Ensure all dataclass validations are comprehensive