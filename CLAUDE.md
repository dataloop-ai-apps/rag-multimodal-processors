# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

**Plan Mode (Default):**
- Start in PLAN mode for every new task
- Gather information and create plans without making changes
- Only move to ACT mode when user types `ACT`
- Do NOT make any changes to code without explicit approval

**Architecture Context:**
When reviewing the code, prioritize reading docs from these directories:
- `chunkers/` - Shared text chunking strategies
- `extractors/` - Shared content extractors (OCR, etc.)
- `pipeline/` - Shared pipeline framework
- `architecture/` - Product requirements and specifications

When adding functionality, ensure any reusable components are placed in shared directories (chunkers, extractors, pipeline, utils) rather than within individual apps.

## Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run PDF processor tests
python -m pytest tests/test_pdf_processor.py

# Run pipeline framework tests
python -m pytest pipeline/tests/

# Run specific test file
python -m pytest pipeline/tests/test_processors.py

# Run with verbose output
python -m pytest -v tests/
```

### Development
```bash
# Install dependencies (project uses .venv virtual environment)
python -m pip install -r requirements.txt  # Note: requirements.txt location varies by app

# Run a single processor locally (requires Dataloop credentials)
# See apps/pdf-processor/dataloop.json for configuration schema
python apps/pdf-processor/pdf_processor.py
```

## Architecture Overview

### Repository Structure

This is a **multi-app monorepo** where each app is independently deployable but shares common modules:

```
rag-multimodal-processors/
├── apps/                          # Independent Dataloop apps (separate deployments)
│   ├── pdf-processor/             # PDF → chunks
│   ├── text-processor/            # .txt, .md, .csv → chunks
│   ├── html-processor/            # HTML → chunks
│   └── email-processor/           # .eml → chunks
├── pipeline/                      # Shared pipeline framework
│   ├── base/processor.py          # BaseProcessor abstract class (4-stage pipeline)
│   ├── config/config_manager.py   # Unified configuration system
│   ├── utils/logging_utils.py     # ProcessorLogger, ErrorHandler
│   └── processor_factory.py       # MIME type → processor mapping
├── chunkers/                      # Shared chunking strategies
│   └── text_chunker.py            # TextChunker with 5 strategies
├── extractors/                    # Shared content extractors
│   └── ocr_extractor.py           # OCRExtractor (EasyOCR + Dataloop models)
└── utils/                         # Shared utilities
    ├── dataloop_helpers.py        # Chunk upload, dataset management
    ├── chunk_metadata.py          # ChunkMetadata class
    └── text_cleaning.py           # Text normalization
```

### Pipeline Pattern

All processors follow a **4-stage Template Method pattern** defined in `pipeline/base/processor.py`:

```python
def process_document(item, target_dataset, context):
    # 1. EXTRACTION: Extract content from file (processor-specific)
    extracted_content = self._extract_content(item, config)

    # 2. PREPROCESSING: Clean/normalize content (uses utils/text_cleaning.py)
    processed_content = self._preprocess_content(extracted_content, config)

    # 3. CHUNKING: Split into chunks (uses chunkers/text_chunker.py)
    chunks = self._chunk_content(processed_content, config)

    # 4. UPLOAD: Store in Dataloop dataset (uses utils/dataloop_helpers.py)
    chunked_items = self._upload_chunks(chunks, item, target_dataset, config)
```

**To add a new processor:**
1. Inherit from `BaseProcessor` in `pipeline/base/processor.py`
2. Implement `_extract_content()` for your file type
3. Register MIME type in `pipeline/processor_factory.py`
4. Common stages (preprocessing, chunking, upload) are handled automatically

### Shared Components

**TextChunker** (`chunkers/text_chunker.py`):
- Strategies: `recursive`, `fixed-size`, `nltk-sentence`, `nltk-paragraphs`, `1-chunk`
- Supports markdown-aware chunking for structured content
- Used by all processors in `_chunk_content()` stage

**OCRExtractor** (`extractors/ocr_extractor.py`):
- Two modes: EasyOCR (local) or Dataloop models (batch processing)
- Caches EasyOCR reader at class level for performance
- Used by PDF processor and potentially other image-containing formats

**ChunkMetadata** (`utils/chunk_metadata.py`):
- Standardized metadata format for all chunks
- Fields: `chunk_index`, `total_chunks`, `source_file`, `processor_type`, etc.
- Applied automatically in `dataloop_helpers.upload_chunks()`

### Configuration System

**Unified config hierarchy** (`pipeline/config/config_manager.py`):
- `BaseConfig` (common): chunking strategy, chunk size, overlap, logging
- Processor-specific configs: `PDFConfig`, `HTMLConfig`, `EmailConfig`, `TextConfig`
- Loaded from JSON files or environment variables
- Validated before processing starts

**Example** (from `apps/pdf-processor/dataloop.json`):
```json
{
  "ocr_from_images": false,
  "use_markdown_extraction": false,
  "chunking_strategy": "recursive",
  "max_chunk_size": 300,
  "chunk_overlap": 50
}
```

### Dataloop Integration

All apps are deployed as **Dataloop Pipeline Nodes**:
- Entry point: `apps/{app-name}/{processor}.py` with class inheriting `dl.BaseServiceRunner`
- Configuration: `apps/{app-name}/dataloop.json` defines UI fields and pipeline interface
- Deployment: Each app has independent Docker image (`gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/{app-name}:{version}`)

**Key patterns:**
- `item.download()` to get file locally → process → `dataset.items.upload()` for chunks
- Configuration retrieved via `context.node.metadata['customNodeConfig']`
- Temporary items/folders cleaned up after OCR processing (see `utils/dataloop_helpers.cleanup_temp_items_and_folder()`)

### Testing Structure

**App-level tests** (`tests/`):
- Integration tests requiring Dataloop credentials
- Use placeholder values: `ITEM_ID`, `TARGET_DATASET_ID`

**Framework tests** (`pipeline/tests/`):
- `test_framework.py`: `ProcessorTestCase` base class with mocks
- `test_processors.py`: Unit tests for all processors
- `test_integration.py`: End-to-end pipeline tests
- Mocks Dataloop API (no credentials needed)

## Key File References

When implementing new features or fixing bugs:

- **PDF extraction logic**: `apps/pdf-processor/pdf_processor.py:138-174` (`_extract_content`)
- **Chunking strategies**: `chunkers/text_chunker.py`
- **OCR implementation**: `extractors/ocr_extractor.py`
- **Chunk upload logic**: `utils/dataloop_helpers.py:upload_chunks()`
- **Base processor template**: `pipeline/base/processor.py:process_document()`
- **Processor factory**: `pipeline/processor_factory.py:create_processor()`

## Common Patterns

### Adding a New Chunking Strategy
1. Add strategy to `chunkers/text_chunker.py:TextChunker.chunk()`
2. Update `strategy` parameter documentation
3. Add to `dataloop.json` configuration options

### Adding OCR Support to a New Processor
1. Import `OCRExtractor` from `extractors/ocr_extractor.py`
2. Extract images in `_extract_content()` method
3. Use `OCRExtractor.extract_text()` or `extract_text_batch()` for processing
4. Combine OCR text with extracted text using desired integration method

### Handling Dataloop Model Execution
1. Inherit from `utils/dataloop_model_executor.py:DataloopModelExecutor`
2. Implement `_process_single_item()` for single-item logic
3. Use `execute_batch()` for batch processing with automatic cleanup
4. See `extractors/ocr_extractor.py` for reference implementation
