# System Architecture: RAG Multimodal Processors

## Overview

A flexible system for processing multimodal documents (PDF, HTML, text, images, etc.) and extracting chunked content for RAG applications using **simple nested function calls**.

## Core Design Principles

1. **Separation of Concerns**: Extractors read files, stages process data, main API orchestrates
2. **Multimodal Support**: Extract and process text, images, tables together
3. **Simple Composition**: Nested function calls instead of complex frameworks
4. **Extensibility**: Easy to add new file types, stages, or processing levels
5. **Type Safety**: Consistent `(data: dict, config: dict) -> dict` signatures

## Architecture

```
┌─────────────────────────────────────┐
│         Main API (main.py)          │
│  - process_item()                   │
│  - process_pdf(), process_docs()    │
│  - 4 built-in levels                │
│  - Custom processing support        │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌─────────────┐
│ Extractors │   │Processing   │
│            │   │Levels       │
│ PDF, HTML, │   │(Functions)  │
│ Docs, Text,│   │             │
│ Email, Img │   │ basic       │
└─────┬──────┘   │ ocr         │
      │          │ llm         │
      │          │ advanced    │
      │          └──────┬──────┘
      │                 │
      └────────┬────────┘
               ▼
       ┌──────────────┐
       │    Stages    │
       │ (Functions)  │
       │              │
       │ preprocess   │
       │ chunk        │
       │ ocr          │
       │ llm          │
       │ upload       │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │   Dataloop   │
       └──────────────┘
```

## Components

### 1. Extractors (`extractors.py`)

**Purpose**: Extract multimodal content from different file types.

**Data Structure**:
```python
@dataclass
class ExtractedContent:
    text: str                       # Main text content
    images: List[ImageContent]      # Extracted images
    tables: List[TableContent]      # Extracted tables
    audio: List[AudioContent]       # Audio content (future)
    metadata: Dict[str, Any]        # File metadata
```

**Available Extractors**:
- `TextExtractor`: `.txt`, `.md`, `.csv` with encoding detection
- `PDFExtractor`: `.pdf` with image extraction and OCR support
- `HTMLExtractor`: `.html`, `.htm` with structure preservation
- `DocsExtractor`: `.docx` (Google Docs) with images and tables
- `EmailExtractor`: `.eml` with headers and body
- `ImageExtractor`: `.png`, `.jpg`, `.jpeg` for OCR processing

**Usage**:
```python
from extractors import get_extractor

extractor = get_extractor('application/pdf')  # Auto-select by MIME type
content = extractor.extract(item, config)

# Returns ExtractedContent with text, images, tables, metadata
```

### 2. Stages (`stages/`)

**Purpose**: Atomic processing operations that transform data.

**Signature**: All stages follow `(data: dict, config: dict) -> dict`

**Categories**:

#### Preprocessing (`stages/preprocessing.py`)
- `clean_text`: Remove extra whitespace, normalize
- `normalize_whitespace`: Standardize spacing
- `remove_empty_lines`: Clean formatting
- `truncate_content`: Limit content length

#### Chunking (`stages/chunking.py`)
- `chunk_recursive`: Hierarchical splitting (default)
- `chunk_by_sentence`: NLTK sentence-based
- `chunk_by_paragraph`: NLTK paragraph-based
- `chunk_fixed_size`: Fixed character chunks
- `no_chunking`: Keep entire document

#### OCR (`stages/ocr.py`)
- `ocr_enhance`: Add OCR text from images
- `describe_images_with_dataloop`: Generate image descriptions using Dataloop vision models
- `ocr_batch_enhance`: Batch OCR processing

#### LLM (`stages/llm.py`)
- `llm_chunk_semantic`: Semantic chunking via Dataloop LLM
- `llm_summarize`: Generate document summaries
- `llm_extract_entities`: Named entity extraction
- `llm_translate`: Translate content

#### Upload (`stages/upload.py`)
- `upload_to_dataloop`: Upload chunks to dataset
- `upload_with_images`: Upload chunks + images
- `upload_with_metadata_only`: Update metadata only
- `dry_run_upload`: Test without uploading

### 3. Processing Levels (`main.py`)

**Purpose**: Pre-built workflows using nested function calls.

**Built-in Levels**:

#### Basic
```python
def basic_processing(data, config):
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

#### OCR
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

#### LLM
```python
def llm_processing(data, config):
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.llm_chunk_semantic(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

#### Advanced
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

## Data Flow

### 1. Extraction Phase
```
Dataloop Item → Extractor → ExtractedContent {
    text: "...",
    images: [...],
    tables: [...],
    metadata: {...}
}
```

### 2. Processing Phase
```
ExtractedContent → to_dict() → {
    'content': str,
    'images': [...],
    'tables': [...],
    'metadata': {...},
    'item': dl.Item,
    'target_dataset': dl.Dataset
}
```

### 3. Stage Execution
```
data → stage1(data, config) → data' → stage2(data', config) → data'' → ...
```

### 4. Upload Phase
```
data['chunks'] → upload_to_dataloop(data, config) → List[dl.Item]
```

## Extension Points

### Adding a New File Type

**Step 1**: Add extractor to `extractors.py`
```python
class AudioExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('audio/mpeg', 'Audio')

    def extract(self, item, config):
        # Transcription logic here
        result = ExtractedContent()
        result.text = transcribed_text
        result.audio = [AudioContent(path=file_path, duration=duration)]
        result.metadata = {'extractor': 'audio'}
        return result
```

**Step 2**: Register
```python
EXTRACTOR_REGISTRY['audio/mpeg'] = AudioExtractor
```

**Step 3**: Use immediately
```python
result = process_item(item, dataset, 'basic')  # Auto-detects MIME type
```

### Adding a New Processing Stage

**Step 1**: Create function in `stages/`
```python
# stages/mystages.py

def my_stage(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """My custom processing"""
    content = data['content']

    # Process content
    processed = my_processing_logic(content, config)

    data['content'] = processed
    data.setdefault('metadata', {})['my_stage_applied'] = True

    return data
```

**Step 2**: Export from `stages/__init__.py`
```python
from .mystages import my_stage
__all__ = [..., 'my_stage']
```

**Step 3**: Use in workflows
```python
workflow = [stages.clean_text, my_stage, stages.upload_to_dataloop]
result = process_custom(item, dataset, workflow)
```

### Adding a Custom Processing Level

**Step 1**: Define level function
```python
def translation_level(data, config):
    """Translate and process"""
    data = stages.clean_text(data, config)
    data = stages.llm_translate(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

**Step 2**: Register
```python
from main import register_processing_level

register_processing_level('translate', translation_level)
```

**Step 3**: Use like built-in levels
```python
result = process_pdf(item, dataset, level='translate',
                    llm_model_id='...', target_language='Spanish')
```

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
    'ocr_integration_method': 'append',  # 'append', 'prepend', 'separate'

    # LLM
    'llm_model_id': 'your-dataloop-model-id',
    'vision_model_id': 'your-vision-model-id',
    'generate_summary': True,
    'extract_entities': False,
    'translate': False,
    'target_language': 'en',

    # Extraction
    'extract_images': True,
    'extract_tables': False,
    'detect_encoding': True,
    'preserve_csv_structure': True,
    'use_markdown_extraction': False,

    # Upload
    'upload_images': False,

    # Preprocessing
    'correct_spelling': False,
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

## Performance Considerations

### Memory Management
- Extractors use `tempfile.TemporaryDirectory()` for automatic cleanup
- Images stored as file paths, not bytes when possible
- Large files should be streamed

### Batch Processing
- Process multiple items using `process_batch()`
- Items processed sequentially (parallel can be added later)
- Error handling per-item to continue processing on failures

### Caching
- OCR extractor caches EasyOCR reader at class level
- Dataloop models cached per session
- Temporary files cleaned up automatically

## Testing Strategy

### Unit Tests
Test individual stages:
```python
def test_clean_text():
    data = {'content': '  Hello  World  '}
    result = clean_text(data, {})
    assert result['content'] == 'Hello World'
```

### Integration Tests
Test complete workflows:
```python
def test_basic_processing():
    # Mock extraction
    data = {
        'content': 'test content',
        'item': mock_item,
        'target_dataset': mock_dataset
    }

    result = basic_processing(data, {})
    assert 'chunks' in result
    assert 'uploaded_items' in result
```

## Future Enhancements

### Planned Features
- Video processing (frame extraction + transcription)
- Audio transcription (speech-to-text)
- Code file processing (syntax-aware chunking)
- JSON/XML structured data processing
- Streaming for large files
- Progress callbacks

### Extensibility Goals
- Plugin system for third-party extractors
- Custom stage registration via decorators
- Visual workflow builder (web UI)
- Performance profiling tools

## Related Documentation

- **[Main README](../README.md)** - Quick start and usage examples
- **[Implementation Summary](../IMPLEMENTATION_SUMMARY.md)** - Quick reference guide
- **[CLAUDE.md](../CLAUDE.md)** - Development guide for Claude Code
- **[Product Requirements](./product_requirements.md)** - Original requirements (historical)
