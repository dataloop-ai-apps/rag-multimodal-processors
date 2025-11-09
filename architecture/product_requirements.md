# Product Requirements: RAG Multimodal Processors

## Overview

The RAG Multimodal Processors system provides a flexible, composable pipeline for processing documents of various types (PDF, HTML, text, images, email, etc.) and extracting chunked content for Retrieval-Augmented Generation (RAG) applications.

The system uses **nested function composition** that separates content extraction and processing stages, enabling flexible and extensible document processing workflows.

## Core Requirements

### 1. Architecture Requirements

#### 1.1 Nested Function Composition
- **Requirement**: System must support composing processing stages using nested function calls
- **Components**:
  - **Extractors**: Extract multimodal content (text, images, tables) from different file types
  - **Stages**: Atomic processing operations (preprocessing, chunking, OCR, LLM, upload)
  - **Processing Levels**: Compose stages into reusable processing workflows
- **Signature**: All stages must follow `(data: dict, config: dict) -> dict`
- **Composition**: Simple nested function calls (e.g., `upload(chunk(clean(data))))`

#### 1.2 Main API
- **Requirement**: Provide simple, high-level API for common use cases
- **Functions**:
  - `process_item()`: Auto-detect file type and process
  - `process_pdf()`, `process_html()`, `process_docs()`, `process_image()`: Type-specific functions
  - `process_batch()`: Process multiple items
  - `process_custom()`: Custom processing sequences
- **Processing Levels**: Four built-in levels (basic, ocr, llm, advanced)

### 2. File Type Support

#### 2.1 Text Files
- **Types**: `.txt`, `.md`, `.csv`
- **Extraction**: Direct text reading with encoding detection
- **Features**:
  - Automatic encoding detection
  - CSV structure preservation
  - Markdown-aware processing

#### 2.2 PDF Files
- **Type**: `.pdf`
- **Extraction**: PyMuPDF and pymupdf4llm support
- **Features**:
  - Text extraction from PDFs
  - Image extraction from pages
  - OCR support for scanned documents
  - Markdown extraction option
  - Table extraction

#### 2.3 HTML Files
- **Types**: `.html`, `.htm`
- **Extraction**: HTML parsing and text extraction
- **Features**:
  - Structure extraction
  - Link parsing
  - Image extraction
  - Table extraction
  - Metadata extraction (title, meta description)

#### 2.4 Word/Docs Files
- **Type**: `.docx` (Google Docs format)
- **Extraction**: Document parsing
- **Features**:
  - Text extraction
  - Image extraction
  - Table extraction
  - Structure preservation

#### 2.5 Email Files
- **Type**: `.eml`
- **Extraction**: Email parsing
- **Features**:
  - Header extraction (sender, recipient, subject, date)
  - Body text extraction
  - Multipart handling
  - Attachment support

#### 2.6 Image Files
- **Types**: `.png`, `.jpg`, `.jpeg`
- **Extraction**: Image loading
- **Features**:
  - OCR text extraction
  - Image description generation
  - Metadata extraction

### 3. Processing Capabilities

#### 3.1 Preprocessing Stages
- **Text Cleaning**: Remove extra whitespace, normalize text
- **Whitespace Normalization**: Standardize spacing
- **Empty Line Removal**: Clean up formatting
- **Content Truncation**: Limit content length

#### 3.2 Chunking Strategies
- **Recursive Chunking**: Hierarchical splitting (default)
- **Sentence-Based**: NLTK sentence-based chunking
- **Paragraph-Based**: NLTK paragraph-based chunking
- **Fixed-Size**: Fixed character chunks
- **No Chunking**: Keep entire document as single chunk
- **LLM Semantic**: Semantic chunking via LLM

#### 3.3 OCR Capabilities
- **EasyOCR**: Local OCR processing
- **Dataloop Models**: Batch OCR processing via Dataloop models
- **Integration Methods**: Append, prepend, or separate OCR text
- **Image Enhancement**: Automatic image preprocessing for OCR

#### 3.4 LLM Capabilities
- **Semantic Chunking**: Chunk documents based on meaning
- **Summarization**: Generate document summaries
- **Entity Extraction**: Named entity extraction
- **Translation**: Translate content to different languages
- **Image Description**: Generate descriptions for images

#### 3.5 Upload Capabilities
- **Standard Upload**: Upload chunks to Dataloop dataset
- **Upload with Images**: Upload chunks with associated images
- **Metadata Updates**: Update metadata without re-uploading
- **Dry Run**: Test processing without uploading

### 4. Processing Levels

#### 4.1 Basic Level
- **Use Case**: Simple text documents
- **Pipeline**: Clean → Normalize → Chunk (Recursive) → Upload
- **Configuration**: `max_chunk_size`, `chunk_overlap`

#### 4.2 OCR Level
- **Use Case**: Documents with images/scans
- **Pipeline**: OCR → Clean → Normalize → Chunk (Recursive) → Upload
- **Configuration**: `use_ocr=True`, `ocr_integration_method`

#### 4.3 LLM Level
- **Use Case**: Semantic chunking
- **Pipeline**: Clean → Normalize → LLM Semantic Chunk → Upload
- **Configuration**: `llm_model_id`

#### 4.4 Advanced Level
- **Use Case**: Full multimodal processing
- **Pipeline**: OCR → Image Descriptions → Clean → Normalize → LLM Chunk → Upload with Images
- **Configuration**: `use_ocr=True`, `llm_model_id`, `vision_model_id`

### 5. Configuration System

#### 5.1 Configuration Structure
Configuration is passed as a dictionary to all stages:

```python
config = {
    # Chunking
    'max_chunk_size': 300,
    'chunk_overlap': 20,
    'chunking_strategy': 'recursive',  # 'recursive', 'fixed-size', 'sentence', 'paragraph', '1-chunk'
    
    # OCR
    'use_ocr': True,
    'ocr_integration_method': 'append',  # 'append', 'prepend', 'separate'
    
    # LLM
    'llm_model_id': 'your-model-id',
    'vision_model_id': 'your-vision-model-id',
    'generate_summary': True,
    'extract_entities': False,
    
    # Extraction
    'extract_images': True,
    'extract_tables': False,
    'use_markdown_extraction': False,  # For PDFs
    'preserve_csv_structure': True,     # For CSVs
    
    # Upload
    'upload_images': False,
    
    # Preprocessing
    'correct_spelling': False,
}
```

#### 5.2 Configuration Requirements
- **Flexibility**: Support per-item and per-batch configuration
- **Validation**: Validate configuration before processing
- **Defaults**: Provide sensible defaults for all options
- **Documentation**: Clear documentation for all configuration options

### 6. Error Handling

#### 6.1 Error Handling Requirements
- **Graceful Degradation**: Handle errors without crashing
- **Error Logging**: Log errors with context (file name, stage, error type)
- **Fallback Mechanisms**: Provide fallbacks for common failure scenarios
- **Input Validation**: Validate file types and sizes before processing

#### 6.2 Error Scenarios
- **Unsupported File Types**: Clear error message with supported types
- **Corrupted Files**: Skip processing with warning
- **Processing Failures**: Log error and return partial results if possible
- **Configuration Errors**: Use defaults with warning
- **Network Failures**: Retry logic for Dataloop API calls

### 7. Performance Requirements

#### 7.1 Processing Time
- **Text files**: < 1 second for files < 1MB
- **HTML files**: < 2 seconds for files < 1MB
- **PDF files**: < 10 seconds for files < 10MB
- **Email files**: < 1 second for files < 1MB
- **Large files**: Complete within 5 minutes for files up to 100MB

#### 7.2 Memory Management
- **Base Memory**: Efficient memory usage for base system
- **File Processing**: Handle files up to 100MB
- **Temporary Files**: Automatic cleanup of temporary files
- **Streaming**: Support streaming for large files where possible

#### 7.3 Concurrency
- **Batch Processing**: Support parallel processing of multiple items
- **Resource Management**: Efficient resource usage during batch processing

### 8. Extensibility Requirements

#### 8.1 Adding New File Types
- **Requirement**: Easy addition of new file type extractors
- **Process**:
  1. Create extractor class inheriting from `BaseExtractor`
  2. Implement `extract()` method
  3. Register in `EXTRACTOR_REGISTRY`
- **Result**: New file type immediately available through main API

#### 8.2 Adding New Processing Stages
- **Requirement**: Easy addition of new processing operations
- **Process**:
  1. Create function with signature `(data: dict, config: dict) -> dict`
  2. Export from `stages/__init__.py`
  3. Use in processing levels via nested function calls
- **Result**: New stage immediately available for composition

#### 8.3 Adding Custom Processing Levels
- **Requirement**: Support for custom processing levels
- **Process**:
  1. Create function that calls stages sequentially with nested functions
  2. Register as processing level using `register_processing_level()`
  3. Use via `process_item()` with custom level name or `process_custom()` with stage list
- **Result**: Custom processing level available for use

### 9. Testing Requirements

#### 9.1 Unit Tests
- **Coverage**: Test individual extractors and stages in isolation
- **Mocking**: Mock external dependencies (Dataloop API, file I/O)
- **Test Cases**: Cover all file types, processing stages, and error scenarios

#### 9.2 Integration Tests
- **Coverage**: Test complete pipelines end-to-end
- **Test Data**: Sample files for each supported file type
- **Scenarios**: Test all processing levels and common configurations

#### 9.3 End-to-End Tests
- **Coverage**: Test with real Dataloop items (requires credentials)
- **Scenarios**: Test actual document processing workflows
- **Validation**: Verify chunks are correctly created and uploaded

### 10. Documentation Requirements

#### 10.1 User Documentation
- **Quick Start Guide**: Simple examples for common use cases
- **API Reference**: Complete documentation of all public functions
- **Configuration Guide**: Detailed explanation of all configuration options
- **Examples**: Code examples for all processing levels and file types

#### 10.2 Developer Documentation
- **Architecture Documentation**: System design and component relationships
- **Extension Guides**: How to add new file types, stages, and pipelines
- **Code Style Guide**: Development guidelines and best practices
- **Testing Guide**: How to write and run tests

### 11. Dependencies

#### 11.1 External Dependencies
- **Dataloop SDK**: For dataset operations and model execution
- **PyMuPDF**: For PDF processing
- **EasyOCR**: For OCR functionality
- **BeautifulSoup4**: For HTML parsing
- **NLTK**: For sentence and paragraph chunking
- **pandas**: For table processing (CSV, tables)

#### 11.2 Python Requirements
- **Python Version**: Python 3.8+
- **Standard Library**: email, tempfile, os, pathlib

### 12. Success Criteria

#### 12.1 Functional Requirements
- ✅ All supported file types process successfully
- ✅ All processing levels work correctly
- ✅ Error handling across all components
- ✅ Configuration system works for all options
- ✅ Extensibility mechanisms work as designed

#### 12.2 Non-Functional Requirements
- ✅ Processing time meets performance requirements
- ✅ Memory usage is efficient
- ✅ Error handling is robust
- ✅ Code is maintainable and well-documented
- ✅ Tests provide adequate coverage

#### 12.3 Usability Requirements
- ✅ Simple API for common use cases
- ✅ Clear error messages
- ✅ Comprehensive documentation
- ✅ Easy to extend with new capabilities

## Related Documentation

- **[System Architecture](./SYSTEM_ARCHITECTURE.md)** - Detailed technical architecture and extension guides
- **[Main README](../README.md)** - Quick start, usage examples, and API reference
- **[CLAUDE.md](../CLAUDE.md)** - Development guide for Claude Code
