# Multi-MIME Type Processor System

A unified pipeline architecture for processing different file types (text, HTML, PDF, email) in RAG workflows.

## 🎯 Overview

This system provides a clean, extensible architecture for processing multiple MIME types through a unified pipeline pattern. All processors share common functionality while maintaining type-specific processing logic.

## 🏗️ Architecture

### Pipeline Pattern
The system implements a 4-stage pipeline:
1. **Extraction**: Extract content from the file
2. **Preprocessing**: Clean and normalize content
3. **Chunking**: Split content into appropriate chunks
4. **Upload**: Store chunks in target dataset

### Base Components
- **BaseProcessor**: Abstract base class for all processors
- **ProcessorFactory**: Factory for creating processors based on MIME type
- **ConfigManager**: Unified configuration system
- **ErrorHandler**: Basic error handling and logging

## 📁 Supported File Types

| File Type | MIME Type | Processor | Features |
|-----------|-----------|-----------|----------|
| Text Files | `text/plain` | TextProcessor | Encoding detection, text cleaning |
| Markdown | `text/markdown` | TextProcessor | Markdown-aware processing |
| CSV Files | `text/csv` | TextProcessor | Structure preservation, metadata |
| HTML Files | `text/html` | HTMLProcessor | Structure extraction, link parsing |
| PDF Files | `application/pdf` | PDFProcessor | Text extraction, OCR, markdown |
| Email Files | `message/rfc822` | EmailProcessor | Header extraction, multipart handling |

## 🚀 Quick Start

### Basic Usage

```python
from pipeline.processor_factory import ProcessorFactory, process_document

# Create processor for specific MIME type
processor = ProcessorFactory.create_processor('text/plain')

# Process document
result = processor.process_document(item, target_dataset, context)
```

### Using the Factory

```python
from pipeline.processor_factory import process_document

# Automatically select processor based on MIME type
result = process_document(item, target_dataset, context)
```

### Custom Configuration

```python
# Configure processing parameters
config = {
    'chunking_strategy': 'recursive',
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'to_correct_spelling': False,
    'ocr_from_images': True,  # PDF specific
    'preserve_structure': True,  # HTML specific
    'extract_headers': True  # Email specific
}
```

## 📦 Installation

### Dependencies

```bash
# Core dependencies
pip install dtlpy
pip install beautifulsoup4
pip install chardet
pip install nltk

# PDF processing
pip install PyMuPDF
pip install pymupdf4llm
pip install easyocr

# Text processing
pip install langchain
```

### Setup

```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## 🔧 Configuration

### Environment Variables

```bash
# Base configuration
export CHUNKING_STRATEGY=recursive
export MAX_CHUNK_SIZE=300
export CHUNK_OVERLAP=20
export LOG_LEVEL=INFO

# Processor-specific configuration
export PDF_OCR_FROM_IMAGES=true
export PDF_USE_MARKDOWN=false
export HTML_PRESERVE_STRUCTURE=true
export EMAIL_INCLUDE_ATTACHMENTS=false
```

### Configuration File

```json
{
  "base": {
    "chunking_strategy": "recursive",
    "max_chunk_size": 300,
    "chunk_overlap": 20,
    "to_correct_spelling": false,
    "enable_logging": true,
    "log_level": "INFO"
  },
  "processors": {
    "pdf": {
      "ocr_from_images": false,
      "use_markdown_extraction": false,
      "ocr_integration_method": "append_to_page"
    },
    "html": {
      "preserve_structure": true,
      "extract_links": true
    },
    "email": {
      "include_attachments": false,
      "extract_headers": true
    },
    "text": {
      "detect_encoding": true,
      "preserve_csv_structure": true
    }
  }
}
```

## 🧪 Testing

### Running Tests

```bash
# Run unit tests
python -m pytest pipeline/tests/test_processors.py -v

# Run integration tests
python -m pytest pipeline/tests/test_integration.py -v

# Run all tests
python -m pytest pipeline/tests/ -v
```

### Test Coverage

```bash
# Generate coverage report
python -m pytest pipeline/tests/ --cov=pipeline --cov-report=html
```

## 📊 Logging

### Log Levels
- **INFO**: General processing information
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures

### Log Format
```
2024-01-01 12:00:00 - text-processor - INFO - Processing text | item_id=123 name=document.txt
2024-01-01 12:00:01 - text-processor - INFO - Extracted content | content_length=1500
2024-01-01 12:00:02 - text-processor - INFO - Created 5 chunks
```

## 🔍 Error Handling

### Graceful Error Handling
- File not found: Skip processing with warning
- Corrupted files: Log error and return empty result
- Processing failures: Log error and continue
- Invalid configuration: Use defaults with warning

### Error Recovery
- Encoding issues: Fallback to UTF-8 with error replacement
- OCR failures: Continue with text-only processing
- Chunking failures: Use single-chunk fallback

## 🚀 Performance

### Benchmarks
- **Text files**: < 1 second for files < 1MB
- **HTML files**: < 2 seconds for files < 1MB
- **PDF files**: < 10 seconds for files < 10MB
- **Email files**: < 1 second for files < 1MB

### Memory Usage
- **Base memory**: ~50MB per processor
- **File processing**: ~100MB per 10MB file
- **OCR processing**: ~200MB additional

## 🔧 Extending the System

### Adding New Processors

```python
from pipeline.base.processor import BaseProcessor

class CustomProcessor(BaseProcessor):
    def __init__(self):
        super().__init__('custom')
    
    def _extract_content(self, item, config):
        # Implement custom extraction logic
        return {'content': content, 'metadata': metadata}
    
    def _get_processor_metadata(self, config):
        # Return custom metadata
        return {'processor_type': 'custom', 'custom_field': 'value'}

# Register processor
ProcessorFactory.register_processor('application/custom', CustomProcessor)
```

### Adding New MIME Types

```python
# Add to processor factory
ProcessorFactory._processors['application/custom'] = CustomProcessor

# Add file extension mapping
extension_mapping = {
    '.custom': 'application/custom'
}
```

## 📚 API Reference

### BaseProcessor

```python
class BaseProcessor(ABC):
    def __init__(self, processor_type: str)
    def process_document(self, item, target_dataset, context) -> List[dl.Item]
    def _extract_content(self, item, config) -> Dict[str, Any]
    def _preprocess_content(self, extracted_content, config) -> Dict[str, Any]
    def _chunk_content(self, processed_content, config) -> List[str]
    def _upload_chunks(self, chunks, original_item, target_dataset, config) -> List[dl.Item]
```

### ProcessorFactory

```python
class ProcessorFactory:
    @classmethod
    def create_processor(cls, mime_type: str) -> Optional[BaseProcessor]
    @classmethod
    def get_supported_mime_types(cls) -> list[str]
    @classmethod
    def is_supported(cls, mime_type: str) -> bool
    @classmethod
    def register_processor(cls, mime_type: str, processor_class: Type[BaseProcessor])
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd rag-multimodal-processors

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the examples in `examples/`

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Support for text, HTML, PDF, and email files
- Unified pipeline architecture
- Basic error handling and logging
- Comprehensive test suite


