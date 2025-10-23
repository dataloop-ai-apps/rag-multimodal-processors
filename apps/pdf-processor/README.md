# PDF Processor

A Dataloop application that extracts text from PDF documents, applies OCR to images, and creates chunks for Retrieval-Augmented Generation (RAG) workflows.

## 🎯 Features

### Text Extraction
- **Plain Text**: Fast extraction using PyMuPDF (fitz)
- **Markdown-Aware**: Preserves document structure (headers, lists, tables) using `pymupdf4llm`

### OCR from Images
Extract embedded images and apply OCR to extract text:

- ✅ Local processing, no upload required
- ✅ Processes images as temporary files
- ✅ No Dataloop items created
- ✅ Fast and efficient text extraction

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

| Parameter | Type | Default | Valid Values/Range | Description |
|-----------|------|---------|-------------------|-------------|
| `ocr_from_images` | boolean | `false` | `true` or `false` | When enabled, extracts all embedded images from the PDF document and applies OCR (Optical Character Recognition) to extract text from them. The extracted text is then integrated with the main document text according to the `ocr_integration_method`. |
| `ocr_integration_method` | string | `"append_to_page"` | `"append_to_page"`, `"separate_chunks"`, or `"combine_all"` | Controls how OCR-extracted text from images is integrated with the PDF text. `append_to_page`: Attaches OCR text directly after the corresponding page text (maintains document structure). `separate_chunks`: Creates a distinct section for all OCR text separate from PDF text. `combine_all`: Appends all OCR text to the end of the document. |
| `use_markdown_extraction` | boolean | `false` | `true` or `false` | When enabled, extracts PDF content in markdown format using `pymupdf4llm`, which preserves document structure such as headers, lists, tables, and formatting. When disabled, uses plain text extraction via PyMuPDF (fitz) which is faster but loses structural information. Enable this for documents where structure is important for chunking and retrieval. |
| `chunking_strategy` | string | `"recursive"` | `"recursive"`, `"fixed-size"`, `"nltk-sentence"`, `"nltk-paragraphs"`, or `"1-chunk"` | Determines how the extracted text is split into chunks. `recursive`: Intelligently splits text respecting semantic boundaries (recommended for most use cases). `fixed-size`: Creates uniform chunks of equal size with overlap. `nltk-sentence`: Splits by sentences using NLTK. `nltk-paragraphs`: Splits by paragraphs using NLTK. `1-chunk`: No splitting, entire document as one chunk (useful for short documents). |
| `max_chunk_size` | integer | `300` | `1` to `2000` characters | Maximum size of each text chunk in characters. Smaller chunks provide more granular retrieval but may lose context. Larger chunks maintain more context but may be less precise. Recommended range: 300-500 for most RAG applications. This parameter works with `chunk_overlap` to control chunk boundaries. |
| `chunk_overlap` | integer | `20` | `0` to `400` characters | Number of characters that overlap between consecutive chunks. Overlap helps maintain context across chunk boundaries and prevents information loss at split points. Should be 10-20% of `max_chunk_size`. Set to 0 for no overlap. Higher values improve context preservation but increase storage requirements. |
| `to_correct_spelling` | boolean | `false` | `true` or `false` | When enabled, applies text cleaning and normalization using the `unstructured.io` library. This includes: unicode quote replacement, non-ASCII character handling, punctuation normalization, and whitespace normalization. Enable for documents with OCR errors or inconsistent formatting. May slow down processing. |

### OCR Configuration

OCR processing uses **EasyOCR** for local, efficient text extraction from images embedded in PDFs.

**To enable OCR:**
```json
{
  "ocr_from_images": true,
  "ocr_integration_method": "append_to_page"
}
```

⚠️ **Note**: OCR processing happens locally and does not require any external model deployment.

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
  "ocr_integration_method": "append_to_page",
  "use_markdown_extraction": true,
  "chunking_strategy": "recursive",
  "max_chunk_size": 500,
  "chunk_overlap": 50,
  "to_correct_spelling": false
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
