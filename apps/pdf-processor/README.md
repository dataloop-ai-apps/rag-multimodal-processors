# PDF Processor

A Dataloop application that extracts text from PDF documents, applies OCR to images, and creates chunks for Retrieval-Augmented Generation (RAG) workflows.

## 🎯 Features

### Text Extraction
- **Plain Text**: Fast extraction using PyMuPDF (fitz)
- **Markdown-Aware**: Preserves document structure (headers, lists, tables) using `pymupdf4llm`

### OCR from Images
Extract embedded images and apply OCR to extract text:

**EasyOCR (Default)**
- ✅ Local processing, no upload required
- ✅ Processes images as temporary files
- ✅ No Dataloop items created

**Custom Dataloop Models**
- ✅ Use any deployed Dataloop OCR model
- ✅ Batch processing for efficiency
- ✅ Automatic cleanup of temporary items
- ✅ Flow: Extract → Upload → Predict → Cleanup

**OCR Integration Methods**
- `append_to_page`: Attaches OCR text to corresponding page (maintains structure)
- `separate_chunks`: Creates distinct sections for PDF text and OCR text
- `combine_all`: Appends all OCR text to the end of PDF text

### Text Chunking
Multiple strategies for optimal embedding:
- `recursive`: Intelligent splitting respecting semantic boundaries
- `fixed-size`: Uniform chunks with configurable overlap
- `nltk-sentence`: Sentence-based chunking
- `nltk-paragraphs`: Paragraph-based chunking
- `1-chunk`: Single chunk (for short documents)

### Text Processing
- **Text Cleaning**: Optional normalization using `unstructured.io`
  - Unicode quote replacement
  - Non-ASCII character handling
  - Punctuation normalization
  - Whitespace normalization

## ⚙️ Configuration

All parameters are configured via the pipeline node in Dataloop.

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ocr_from_images` | boolean | `false` | Extract images from PDF and apply OCR |
| `custom_ocr_model_id` | string | `null` | Deployed Dataloop OCR model ID (optional, uses EasyOCR if not provided) |
| `ocr_integration_method` | string | `"append_to_page"` | How to integrate OCR text |
| `use_markdown_extraction` | boolean | `false` | Preserve document structure using markdown |
| `chunking_strategy` | string | `"recursive"` | Text chunking strategy |
| `max_chunk_size` | integer | `300` | Maximum chunk size in characters |
| `chunk_overlap` | integer | `20` | Overlap between chunks in characters |
| `to_correct_spelling` | boolean | `false` | Apply text cleaning and normalization |
| `remote_path_for_chunks` | string | `"/chunks"` | Remote path for storing chunks |
| `target_dataset` | string | `null` | Target dataset ID (auto-creates if not specified) |

### OCR Configuration

**To use EasyOCR (default):**
```json
{
  "ocr_from_images": true,
  "custom_ocr_model_id": null
}
```

**To use a custom Dataloop OCR model:**
```json
{
  "ocr_from_images": true,
  "custom_ocr_model_id": "your-model-id-here"
}
```
⚠️ **Note**: Custom model must be deployed before use.

## 🚀 Deployment

1. Go to Dataloop Marketplace
2. Search for "PDF to Chunks"
3. Click "Install"
4. Configure in your pipeline

## 📊 Usage

### In a Pipeline

1. **Add the app node** to your pipeline
2. **Configure parameters** in the node settings
3. **Connect input** (PDF items from dataset)
4. **Connect output** (chunks will be created in target dataset)
5. **Run the pipeline**

### Example Pipeline Flow

```
[PDF Dataset] → [PDF to Chunks] → [Embedding Model] → [Vector Database]
```

### Example Configuration for Production

```json
{
  "ocr_from_images": true,
  "custom_ocr_model_id": null,
  "ocr_integration_method": "append_to_page",
  "use_markdown_extraction": true,
  "chunking_strategy": "recursive",
  "max_chunk_size": 500,
  "chunk_overlap": 50,
  "to_correct_spelling": false,
  "remote_path_for_chunks": "/chunks",
  "target_dataset": null
}
```

## 📤 Output Format

### Chunk Items

Each chunk is uploaded as a text file (`.txt`):
```
{original_filename}_chunk_{index:04d}.txt
```

Example:
```
document.pdf → document_chunk_0001.txt
             → document_chunk_0002.txt
             → document_chunk_0003.txt
```

### Metadata Structure

Each chunk includes comprehensive metadata for provenance tracking:

```json
{
  "user": {
    "document": "example.pdf",
    "document_type": "application/pdf",
    "total_pages": 25,
    "total_chunks": 120,
    "extraction_method": "pymupdf4llm",
    "extraction_format": "markdown",
    "chunking_strategy": "recursive",
    "markdown_aware_splitting": true,
    "extracted_chunk": true,
    "original_item_id": "65f2a3b4c1e2d3f4a5b6c7d8",
    "original_dataset_id": "65f2a3b4c1e2d3f4a5b6c7d9",
    "processing_timestamp": 1698765432.123
  }
}
```
