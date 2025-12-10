# RAG Multimodal Processors

A collection of independent Dataloop applications for processing various file types and creating chunks for Retrieval-Augmented Generation (RAG) workflows.

## 🎯 Overview

This repository contains **separate apps per data type**, each independently deployable as a Dataloop service. All apps share common modules (chunkers, utils, extractors) to avoid code duplication while maintaining independence.

## 📁 Repository Structure

```
rag-multimodal-processors/
├── apps/                    # Independent applications
│   ├── pdf-processor/      # PDF processing app
│   ├── image-processor/    # Coming soon
│   └── audio-processor/    # Coming soon
├── chunkers/               # Shared chunking strategies
├── utils/                  # Shared utilities
├── extractors/             # Shared extractors (OCR, transcription, etc.)
└── tests/                  # Centralized tests
```

### Key Design Principles

1. **App Independence**: Each app is a separate service with its own deployment
2. **Shared Modules**: Common functionality (chunkers, utils, extractors) shared across apps
3. **Dataloop Model Integration**: Support for Dataloop models for OCR, transcription, etc.
4. **External Library Fallbacks**: EasyOCR, Whisper, and other libraries when no model provided
5. **Flexible Configuration**: Each app has its own config schema in `dataloop.json`
6. **Easy Deployment**: Independent Docker images and versioning per app

## 📦 Available Apps

### 📄 [PDF Processor](apps/pdf-processor/)
Processes PDF documents to extract text, apply OCR, and create chunks for RAG.

**Features:**
- Text extraction (plain and markdown-aware)
- Image extraction and OCR (EasyOCR or custom models)
- Multiple chunking strategies
- Configurable text cleaning

**[→ Full Documentation](apps/pdf-processor/README.md)**

### 📝 Document Processor *(Coming Soon)*
Process Word documents, text files, and other document formats for RAG workflows.

### 🖼️ Image Processor *(Coming Soon)*
Process images for RAG workflows with captioning and OCR.

### 🎵 Audio Processor *(Coming Soon)*
Transcribe audio files and create searchable chunks.

### 🎬 Video Processor *(Coming Soon)*
Extract audio, captions, and keyframes from videos.

## 🧩 Shared Modules

### Chunkers (`chunkers/`)

Reusable text chunking strategies for breaking content into embedding-friendly pieces:

- **TextChunker**: General text chunking with multiple strategies
  - `recursive`: Intelligent splitting respecting semantic boundaries
  - `fixed-size`: Uniform chunks with configurable overlap
  - `nltk-sentence`: Sentence-based chunking
  - `nltk-paragraphs`: Paragraph-based chunking
  - `markdown-aware`: Respects markdown structure

**Future Chunkers:**
- `CodeChunker`: Function/class-based chunking for code
- `TableChunker`: Row-wise chunking with header preservation
- `StructuredChunker`: Hierarchy-preserving for JSON/XML

### Extractors (`extractors/`)

Unified interfaces for content extraction using Dataloop models or external libraries:

- **OCRExtractor**: Text extraction from images
  - EasyOCR (default): Local processing
  - Custom Dataloop models: Batch processing with automatic cleanup

**Future Extractors:**
- `TranscriptionExtractor`: Audio → Text
- `CaptioningExtractor`: Image → Description
- `EmbeddingExtractor`: Content → Vectors

### Utils (`utils/`)

Shared utility functions:

- **`text_cleaning.py`**: Text normalization and cleaning
- **`dataloop_helpers.py`**: Dataset operations, chunk uploading, metadata handling
- **`dataloop_model_executor.py`**: Base class for Dataloop model execution

## 🔗 Links

- [Dataloop Platform](https://dataloop.ai)
- [Dataloop SDK Documentation](https://sdk-docs.dataloop.ai)
