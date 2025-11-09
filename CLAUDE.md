# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

**Plan Mode (Default):**
- Start in PLAN mode for every new task
- Gather information and create plans without making changes
- Only move to ACT mode when user types `ACT`
- Do NOT make any changes to code without explicit approval

## Architecture Overview

The system uses **nested function composition** with three main components:

1. **Extractors** (`extractors.py`) - Extract multimodal content from different file types
2. **Stages** (`stages/`) - Atomic processing operations with consistent signatures
3. **Main API** (`main.py`) - Orchestration with built-in processing levels

**Architecture Context:**
When reviewing the code, prioritize reading docs from these directories:
- `main.py` - Main API and orchestrator with 4 built-in processing levels
- `extractors.py` - File type extractors (PDF, HTML, .docx, text, images, email)
- `stages/` - Processing stages (preprocessing, chunking, OCR, LLM, upload)
- `chunkers/` - Shared text chunking implementations
- `extractors/` (directory) - Shared extractors (OCR)
- `utils/` - Shared utilities (dataloop_helpers, chunk_metadata, text_cleaning)
- `architecture/` - Product requirements and specifications

When adding functionality, ensure any reusable components follow the appropriate pattern:
- New file types → Add to `extractors.py` and register in `EXTRACTOR_REGISTRY`
- New processing operations → Add to `stages/` with signature `(data: dict, config: dict) -> dict`
- New processing levels → Use `register_processing_level()` in `main.py`

## Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v
```

### Development
```bash
# Install dependencies (project uses .venv virtual environment)
python -m pip install -r requirements.txt

# Example usage (requires Dataloop credentials)
python main.py
```

## Repository Structure

```
rag-multimodal-processors/
├── main.py                      # Main API with processing levels
├── extractors.py                # Multimodal extractors (PDF, HTML, docs, text, images, email)
├── stages/                      # Processing stages (consistent signatures)
│   ├── preprocessing.py         # clean_text, normalize_whitespace
│   ├── chunking.py              # chunk_recursive, chunk_by_sentence, etc.
│   ├── ocr.py                   # ocr_enhance, describe_images_with_dataloop
│   ├── llm.py                   # llm_chunk_semantic, llm_summarize, llm_translate
│   └── upload.py                # upload_to_dataloop, upload_with_images
├── chunkers/                    # Shared chunking implementations
│   └── text_chunker.py          # TextChunker with 5 strategies
├── extractors/ (dir)            # Shared extractors (OCR)
│   └── ocr_extractor.py         # OCRExtractor (EasyOCR + Dataloop models)
├── utils/                       # Shared utilities
│   ├── dataloop_helpers.py      # Chunk upload, dataset management
│   ├── chunk_metadata.py        # ChunkMetadata class
│   └── text_cleaning.py         # Text normalization
└── architecture/                # Documentation
    ├── SYSTEM_ARCHITECTURE.md   # Detailed architecture
    └── product_requirements.md  # Product requirements
```

## Core Design Principles

1. **Simple Nested Functions**: No piping operators or complex frameworks
2. **Consistent Signatures**: All stages follow `(data: dict, config: dict) -> dict`
3. **Multimodal Extraction**: Text, images, tables in one pass
4. **Four Processing Levels**: basic, ocr, llm, advanced
5. **Easy to Extend**: Add extractors, stages, or levels with minimal code

## Processing Levels

The system provides four built-in processing levels using nested function calls:

### Basic Processing
```python
def basic_processing(data, config):
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

### OCR Processing
```python
def ocr_processing(data, config):
    config['use_ocr'] = True
    data = stages.ocr_enhance(data, config)
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

### LLM Processing
```python
def llm_processing(data, config):
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.llm_chunk_semantic(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

### Advanced Processing
```python
def advanced_processing(data, config):
    config['use_ocr'] = True
    config['describe_images'] = True
    data = stages.ocr_enhance(data, config)
    data = stages.describe_images_with_dataloop(data, config)
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.llm_chunk_semantic(data, config)
    data = stages.upload_with_images(data, config)
    return data
```

## Key File References

When implementing new features or fixing bugs:

- **Main orchestration**: `main.py` - Processing levels and convenience functions
- **Content extraction**: `extractors.py` - All file type extractors
- **Processing stages**: `stages/` - All processing operations
- **Chunking strategies**: `chunkers/text_chunker.py`
- **OCR implementation**: `extractors/ocr_extractor.py`
- **Chunk upload logic**: `utils/dataloop_helpers.py:upload_chunks()`
- **Architecture documentation**: `architecture/SYSTEM_ARCHITECTURE.md`

## Common Patterns

### Adding a New File Type
1. Add extractor to `extractors.py`:
```python
class AudioExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('audio/mpeg', 'Audio')

    def extract(self, item, config):
        result = ExtractedContent()
        result.text = "transcribed text"
        result.audio = [AudioContent(path=file_path, duration=duration)]
        result.metadata = {'extractor': 'audio'}
        return result
```

2. Register in `EXTRACTOR_REGISTRY`:
```python
EXTRACTOR_REGISTRY['audio/mpeg'] = AudioExtractor
```

3. Use immediately:
```python
result = process_item(item, dataset, 'basic')  # Auto-detects MIME type
```

### Adding a New Processing Stage
1. Create function in `stages/`:
```python
def my_stage(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """My custom processing"""
    data['content'] = transform(data['content'])
    data.setdefault('metadata', {})['my_stage_applied'] = True
    return data
```

2. Export from `stages/__init__.py`:
```python
from .mystages import my_stage
__all__ = [..., 'my_stage']
```

3. Use in workflows:
```python
custom_workflow = [stages.clean_text, my_stage, stages.upload_to_dataloop]
result = process_custom(item, dataset, custom_workflow)
```

### Adding a Custom Processing Level
1. Define level function:
```python
def translation_level(data, config):
    data = stages.clean_text(data, config)
    data = stages.llm_translate(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

2. Register:
```python
from main import register_processing_level
register_processing_level('translate', translation_level)
```

3. Use like built-in levels:
```python
result = process_pdf(item, dataset, level='translate',
                    llm_model_id='...', target_language='Spanish')
```

### Custom Workflow
```python
from main import process_custom
import stages

custom_workflow = [
    stages.ocr_enhance,
    stages.clean_text,
    stages.chunk_by_sentence,
    stages.upload_to_dataloop
]

result = process_custom(item, dataset, custom_workflow, {'use_ocr': True})
```

## Dataloop Integration

All processing uses **Dataloop items directly**:
- `item.download()` to get file locally → process → `dataset.items.upload()` for chunks
- Configuration retrieved via context or passed as dict
- All LLM/vision processing uses Dataloop models (no external APIs)
- Temporary items/folders cleaned up after processing

## Testing Structure

**Tests** (`tests/`):
- Integration tests requiring Dataloop credentials
- Use placeholder values: `ITEM_ID`, `TARGET_DATASET_ID`
- Test individual stages and complete processing levels

## Configuration System

Configuration is passed as a dictionary to all stages:

```python
config = {
    # Chunking
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'chunking_strategy': 'recursive',

    # OCR
    'use_ocr': True,
    'ocr_integration_method': 'append',

    # LLM
    'llm_model_id': 'your-dataloop-model-id',
    'vision_model_id': 'your-vision-model-id',
    'generate_summary': True,

    # Extraction
    'extract_images': True,
    'extract_tables': False,
    'use_markdown_extraction': False,

    # Upload
    'upload_images': False,
}
```

## Error Handling

### Stage-Level
Stages handle errors gracefully:
```python
def my_stage(data, config):
    try:
        # Process
        return data
    except Exception as e:
        print(f"Warning: {e}")
        return data  # Return unchanged on error
```

### Processing-Level
Main API handles extraction and orchestration errors:
```python
try:
    result = process_item(item, dataset, 'ocr')
except ValueError as e:
    # Invalid processing level or unsupported MIME type
    print(f"Error: {e}")
```

## Why Nested Functions?

1. **Simplicity**: Easy to read and understand
2. **Explicit**: Clear execution order
3. **Flexible**: Easy to create custom sequences
4. **No magic**: No operator overloading or hidden behavior
5. **Standard Python**: Uses familiar patterns

Compare:
```python
# Nested functions (current approach)
data = stages.clean_text(data, config)
data = stages.chunk_recursive(data, config)
data = stages.upload_to_dataloop(data, config)

# vs. Complex piping (avoided)
data = data | clean_text | chunk_recursive | upload_to_dataloop
```

The nested function approach is more explicit and easier to debug.
