# Product Requirements: Multi-MIME Type Processor Refactoring

## Overview

This document outlines the requirements for refactoring the RAG Multimodal Processors to support multiple MIME types (text, .eml, PDF, HTML) through a unified pipeline architecture. The refactoring will create a clean, extensible system that handles different file types with consistent processing patterns, error handling, and logging.

## Current State Analysis

The current system has:
- **PDF Processor**: Fully implemented with text extraction, OCR, and chunking
- **HTML Processor**: Empty directory (needs implementation)
- **Text Processor**: Empty directory (needs implementation)
- **Shared Modules**: TextChunker, OCRExtractor, and utility functions

## Requirements

### 1. Core Architecture

#### 1.1 Unified Processor Interface
- **Requirement**: Create a base `Processor` class that all MIME-specific processors inherit from
- **Interface**: Standardized `process_document(item, target_dataset, context)` method
- **Configuration**: Unified configuration schema across all processors
- **Logging**: Consistent logging format with processor type identification

#### 1.2 Pipeline Pattern Implementation
- **Requirement**: Implement a pipeline pattern for document processing
- **Stages**: 
  1. **Extraction**: Extract content from the file
  2. **Preprocessing**: Clean and normalize content
  3. **Chunking**: Split content into appropriate chunks
  4. **Upload**: Store chunks in target dataset
- **Piping**: Each stage passes data to the next stage with error handling

### 2. MIME Type Support

#### 2.1 Text Files (.txt, .md, .csv)
- **Extraction**: Direct text reading with encoding detection
- **Preprocessing**: Text cleaning and normalization
- **Chunking**: Use existing TextChunker with appropriate strategies
- **Special Handling**: CSV files should preserve structure in metadata

#### 2.2 Email Files (.eml)
- **Extraction**: Parse email headers, body, and attachments
- **Preprocessing**: Extract sender, recipient, subject, date, and body text
- **Chunking**: Separate chunks for headers and body content
- **Metadata**: Include email-specific metadata (sender, subject, date)

#### 2.3 PDF Files (.pdf)
- **Extraction**: Maintain existing PyMuPDF and pymupdf4llm support
- **OCR**: Keep existing OCR functionality (EasyOCR and Dataloop models)
- **Chunking**: Preserve existing chunking strategies
- **Enhancement**: Add better error handling and fallback mechanisms

#### 2.4 HTML Files (.html, .htm)
- **Extraction**: Parse HTML structure and extract text content
- **Preprocessing**: Remove HTML tags, preserve semantic structure
- **Chunking**: Use markdown-aware chunking for structured content
- **Metadata**: Include page title, meta description, and link information

### 3. Error Handling & Resilience

#### 3.1 Basic Error Handling
- **Requirement**: All processors must handle errors gracefully without crashing
- **Simple Fallbacks**: 
  - If processing fails, log error and return empty result
  - If file is corrupted, skip processing with warning
- **Error Logging**: Basic error logs with file name and error type
- **Input Validation**: Basic file type and size checks

### 4. Testing Requirements

#### 4.1 Basic Unit Tests
- **Coverage**: Basic test coverage for core functionality
- **Test Cases**: 
  - Valid file processing for each MIME type
  - Basic error handling scenarios
- **Mocking**: Mock external dependencies (Dataloop API)

#### 4.2 Simple Integration Tests
- **End-to-End**: Basic processing pipeline tests
- **Test Data**: Sample files for each MIME type

### 5. Basic Logging

#### 5.1 Simple Logging
- **Format**: Basic text logs with key information
- **Levels**: INFO, WARNING, ERROR
- **Context**: Include processor type and file name
- **Performance**: Log basic processing times

### 6. Configuration Management

#### 6.1 Unified Configuration Schema
- **Base Config**: Common settings (chunking, logging, error handling)
- **MIME-Specific Config**: Processor-specific settings
- **Environment Variables**: Support for environment-based configuration
- **Validation**: Configuration validation with clear error messages

#### 6.2 Configuration Examples
```json
{
  "base": {
    "chunking_strategy": "recursive",
    "max_chunk_size": 300,
    "chunk_overlap": 20,
    "enable_logging": true,
    "log_level": "INFO"
  },
  "processors": {
    "pdf": {
      "ocr_from_images": false,
      "use_markdown_extraction": false
    },
    "html": {
      "preserve_structure": true,
      "extract_links": true
    },
    "eml": {
      "include_attachments": false,
      "extract_headers": true
    }
  }
}
```

### 7. Implementation Plan

#### Phase 1: Base Architecture (Week 1-2)
1. Create base `Processor` class and pipeline framework
2. Implement unified configuration system
3. Set up structured logging and error handling
4. Create comprehensive test framework

#### Phase 2: Core Processors (Week 3-4)
1. Refactor PDF processor to use new architecture
2. Implement text processor (.txt, .md, .csv)
3. Implement HTML processor
4. Add unit tests for all processors

#### Phase 3: Advanced Features (Week 5-6)
1. Implement email processor (.eml)
2. Add performance optimizations
3. Implement monitoring and metrics
4. Add integration tests

#### Phase 4: Documentation & Deployment (Week 7-8)
1. Update documentation and README files
2. Create deployment guides
3. Performance testing and optimization
4. Final testing and bug fixes

### 8. Success Criteria

#### 8.1 Functional Requirements
- ✅ All MIME types (text, .eml, PDF, HTML) process successfully
- ✅ Basic error handling across all processors
- ✅ Basic test coverage for core functionality
- ✅ Simple logging with key information

#### 8.2 Non-Functional Requirements
- ✅ Processing time < 30 seconds for files < 10MB
- ✅ Basic error handling for corrupted files

#### 8.3 Maintainability Requirements
- ✅ Clear separation of concerns
- ✅ Extensible architecture for new MIME types
- ✅ Basic documentation

### 9. Dependencies & Constraints

#### 9.1 External Dependencies
- **Dataloop SDK**: For dataset operations and model execution
- **PyMuPDF**: For PDF processing
- **EasyOCR**: For OCR functionality
- **BeautifulSoup4**: For HTML parsing
- **email**: Python standard library for .eml processing

#### 9.2 Performance Constraints
- **Memory**: Must handle files up to 100MB
- **Processing Time**: Must complete within 5 minutes for large files
- **Concurrency**: Support for parallel processing of multiple files
- **Storage**: Efficient temporary file management

### 10. Risk Mitigation

#### 10.1 Technical Risks
- **Memory Leaks**: Basic resource cleanup
- **API Failures**: Simple error handling and logging

#### 10.2 Operational Risks
- **Configuration Errors**: Basic validation and error messages 