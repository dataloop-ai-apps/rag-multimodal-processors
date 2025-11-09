# RAG Multimodal Processors

A flexible, composable system for processing multimodal documents (PDF, HTML, images, text, etc.) and extracting chunked content for Retrieval-Augmented Generation (RAG) workflows.

## 🎯 Overview

**Modular document processing with support for:**

- **Multiple file types**: PDF, HTML, .docx, text, images, email
- **Multimodal extraction**: Text, images, tables from documents
- **Flexible processing**: OCR, LLM-based chunking, semantic analysis
- **Simple composition**: Nested function calls for custom workflows
- **Dataloop native**: Direct integration with Dataloop items and models

## 🚀 Quick Start

### Basic Usage

```python
import dtlpy as dl
from main import process_pdf

# Get item and dataset
item = dl.items.get(item_id='your-item-id')
dataset = dl.datasets.get(dataset_id='your-dataset-id')

# Process with a built-in level
result = process_pdf(item, dataset, level='basic')
print(f"Created {len(result)} chunks")
```

### Processing Levels

Four built-in processing levels:

```python
# 1. Basic: Clean → Chunk → Upload
result = process_pdf(item, dataset, level='basic')

# 2. OCR: OCR → Clean → Chunk → Upload
result = process_pdf(item, dataset, level='ocr', use_ocr=True)

# 3. LLM: Clean → LLM Semantic Chunk → Upload
result = process_pdf(item, dataset, level='llm', llm_model_id='...')

# 4. Advanced: OCR → Image Descriptions → LLM Chunk → Upload
result = process_pdf(item, dataset, level='advanced',
                    use_ocr=True,
                    llm_model_id='...',
                    vision_model_id='...')
```

### Custom Processing

```python
from main import process_custom
import stages

# Define custom workflow
custom_workflow = [
    stages.ocr_enhance,
    stages.clean_text,
    stages.chunk_by_sentence,
    stages.upload_to_dataloop
]

result = process_custom(item, dataset, custom_workflow, {'use_ocr': True})
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Main API (main.py)          │
│  process_pdf(), process_docs(), etc.│
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌─────────────┐
│ Extractors │   │Processing   │
│(File Types)│   │Levels       │
└─────┬──────┘   └──────┬──────┘
      │                 │
      └────────┬────────┘
               ▼
       ┌──────────────┐
       │    Stages    │
       │(Nested Calls)│
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │   Dataloop   │
       └──────────────┘
```

### Key Components

1. **Extractors** (`extractors.py`) - Extract content from file types (PDF, HTML, .docx, text, images, email)
2. **Stages** (`stages/`) - Processing operations (clean, chunk, OCR, LLM, upload)
3. **Main API** (`main.py`) - Orchestration with 4 built-in levels + custom processing

## 📦 Core Components

### Extractors

Extract multimodal content from files:

```python
from extractors import get_extractor

extractor = get_extractor('application/pdf')
content = extractor.extract(item, config)

# Returns: text, images, tables, metadata
```

**Supported File Types**:
- Text: `.txt`, `.md`, `.csv`
- PDF: `.pdf` (with images)
- HTML: `.html`, `.htm`
- Docs: `.docx` (Google Docs)
- Email: `.eml`
- Images: `.png`, `.jpg`, `.jpeg`

### Stages

Processing functions with signature: `(data: dict, config: dict) -> dict`

```python
import stages

# Preprocessing
stages.clean_text
stages.normalize_whitespace

# Chunking
stages.chunk_recursive
stages.chunk_by_sentence
stages.chunk_by_paragraph

# OCR
stages.ocr_enhance
stages.describe_images_with_dataloop

# LLM
stages.llm_chunk_semantic
stages.llm_summarize
stages.llm_translate

# Upload
stages.upload_to_dataloop
stages.upload_with_images
```

### Processing Levels

Built-in workflows using nested function calls:

```python
def basic_processing(data, config):
    """Clean → Chunk → Upload"""
    data = stages.clean_text(data, config)
    data = stages.normalize_whitespace(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data
```

## 🔧 Custom Processing

### Option 1: Custom Workflow

```python
from main import process_custom
import stages

workflow = [
    stages.ocr_enhance,
    stages.clean_text,
    stages.llm_chunk_semantic,
    stages.upload_with_images
]

result = process_custom(item, dataset, workflow, {
    'use_ocr': True,
    'llm_model_id': 'your-model-id'
})
```

### Option 2: Register Custom Level

```python
from main import register_processing_level
import stages

def translation_level(data, config):
    data = stages.clean_text(data, config)
    data = stages.llm_translate(data, config)
    data = stages.chunk_recursive(data, config)
    data = stages.upload_to_dataloop(data, config)
    return data

# Register
register_processing_level('translate', translation_level)

# Use
result = process_pdf(item, dataset, level='translate',
                    llm_model_id='...', target_language='Spanish')
```

### Option 3: Custom Stage

```python
def my_stage(data, config):
    """Custom processing stage"""
    data['content'] = transform(data['content'])
    return data

# Use in workflow
workflow = [stages.clean_text, my_stage, stages.upload_to_dataloop]
result = process_custom(item, dataset, workflow)
```

## ⚙️ Configuration

```python
config = {
    # Chunking
    'max_chunk_size': 300,
    'chunk_overlap': 20,

    # OCR
    'use_ocr': True,
    'ocr_integration_method': 'append',  # or 'prepend', 'separate'

    # LLM
    'llm_model_id': 'your-dataloop-model-id',
    'vision_model_id': 'your-vision-model-id',
    'generate_summary': True,

    # Extraction
    'extract_images': True,
    'extract_tables': False,
    'use_markdown_extraction': False,  # For PDFs

    # Upload
    'upload_images': False,
}
```

## 🧪 Examples

### Basic PDF Processing

```python
from main import process_pdf

result = process_pdf(item, dataset, level='basic',
                    max_chunk_size=500, chunk_overlap=50)
```

### PDF with OCR

```python
result = process_pdf(item, dataset, level='ocr',
                    use_ocr=True, ocr_integration_method='append')
```

### LLM Semantic Chunking

```python
result = process_pdf(item, dataset, level='llm',
                    llm_model_id='your-model-id')
```

### Full Multimodal

```python
result = process_pdf(item, dataset, level='advanced',
                    use_ocr=True,
                    llm_model_id='your-llm-model',
                    vision_model_id='your-vision-model',
                    generate_summary=True,
                    upload_images=True)
```

### Translation Workflow

```python
from main import process_custom
import stages

workflow = [
    stages.clean_text,
    stages.llm_translate,
    stages.chunk_recursive,
    stages.upload_to_dataloop
]

result = process_custom(item, dataset, workflow, {
    'llm_model_id': 'your-model-id',
    'translate': True,
    'target_language': 'Spanish'
})
```

### Batch Processing

```python
from main import process_batch

items = dataset.items.list()
results = process_batch(items, target_dataset, 'ocr',
                       config={'use_ocr': True}, verbose=True)
```

## 🔌 Extending the System

### Add New File Type

Add to `extractors.py`:

```python
class AudioExtractor(BaseExtractor):
    def __init__(self):
        super().__init__('audio/mpeg', 'Audio')

    def extract(self, item, config):
        # Extraction logic
        result = ExtractedContent()
        result.text = "..."
        return result

# Register
EXTRACTOR_REGISTRY['audio/mpeg'] = AudioExtractor
```

### Add New Stage

Create in `stages/`:

```python
def my_stage(data, config):
    """My processing logic"""
    data['content'] = process(data['content'])
    return data
```

Export from `stages/__init__.py` and use immediately.

### Add New Level

```python
from main import register_processing_level
import stages

def my_level(data, config):
    data = stages.stage1(data, config)
    data = stages.stage2(data, config)
    return data

register_processing_level('my_level', my_level)
```

## 📊 Supported File Types

| File Type | Extractor | Features |
|-----------|-----------|----------|
| Text (.txt, .md, .csv) | TextExtractor | Encoding detection, CSV structure |
| PDF | PDFExtractor | Text + images + OCR + markdown |
| HTML | HTMLExtractor | Text + images + tables |
| Word/Docs (.docx) | DocsExtractor | Text + images + tables |
| Email (.eml) | EmailExtractor | Headers + body |
| Images (.png, .jpg) | ImageExtractor | OCR processing |

## 📚 Documentation

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Quick reference guide
- **[SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md)** - Detailed architecture
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code

## 🔗 Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK Documentation](https://sdk-docs.dataloop.ai)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v
```

## 🤝 Contributing

```bash
# Clone and setup
git clone <repository-url>
cd rag-multimodal-processors
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

**Code Style**:
- Stage signature: `(data: dict, config: dict) -> dict`
- Use type hints
- Add docstrings
- Write tests

## 📄 License

MIT License

## 🆘 Support

- Create an issue in the repository
- Check [documentation](architecture/)
- Review [examples](#-examples)
