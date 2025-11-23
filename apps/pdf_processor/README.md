# PDF Processor

A modular Dataloop application for processing PDF files into RAG-ready chunks with ML-enhanced extraction and flexible OCR.

## 🎯 Features

### Text Extraction
- **Plain Text**: Standard extraction using PyMuPDF (fitz)
- **ML-Enhanced Markdown**: PyMuPDF Layout for improved structure preservation
  - ML-based layout analysis
  - Intelligent OCR evaluation (uses Tesseract when beneficial)
  - Better header/footer detection
  - Improved multi-column and table handling

### Chunk Upload
- **Bulk Upload**: Upload all chunks in a single operation using pandas DataFrame
- **Per-Chunk Metadata**: Track page numbers, images, extraction method per chunk
- **Resilient Fallback**: Automatic retry with individual uploads if needed

### OCR from Images
Extract embedded images and apply OCR to extract text:

- **Flexible OCR Backends**: Dataloop models, EasyOCR fallback, or Tesseract via PyMuPDF Layout
- **Local Processing**: Images processed as temporary files without creating Dataloop items
- **Multiple Integration Methods**: Choose how OCR text integrates with document text

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
| `correct_spelling` | boolean | `false` | `true` or `false` | When enabled, applies text cleaning and normalization using the `unstructured.io` library. This includes: unicode quote replacement, non-ASCII character handling, punctuation normalization, and whitespace normalization. Enable for documents with OCR errors or inconsistent formatting. May slow down processing. |

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
  "correct_spelling": false
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
    "source_item_id": "65f2a3b4c1e2d3f4a5b6c7d8",
    "source_file": "example.pdf",
    "source_dataset_id": "65f2a3b4c1e2d3f4a5b6c7d9",
    "chunk_index": 0,
    "total_chunks": 120,
    "extracted_chunk": true,
    "processing_timestamp": 1698765432.123,
    "processor": "pdf",
    "extraction_method": "pymupdf4llm_layout",
    "layout_enhancement": true,
    "page_numbers": [1, 2],
    "image_ids": ["img_id_1", "img_id_2"]
  }
}
```

**Key Metadata Fields:**
- `extraction_method`: `"pymupdf4llm_layout"` (ML-enhanced) or `"pymupdf4llm"` (standard) or `"pymupdf"` (plain text)
- `layout_enhancement`: `true` if PyMuPDF Layout was active during extraction
- `page_numbers`: List of source pages for this chunk
- `image_ids`: IDs of associated images (if any)
- `chunk_index` / `total_chunks`: Position in document for reconstruction

## 🏗️ Architecture

The PDF processor uses a type-safe, stateless architecture:

```
PDFProcessor (app.py)
    ├── PDFExtractor (pdf_extractor.py) - PDF-specific extraction
    └── Transforms - Shared pipeline operations
        ├── transforms.clean() - Text normalization
        ├── transforms.chunk() - Text chunking
        └── transforms.upload_to_dataloop() - Chunk upload
```

**Key Components:**
- `ExtractedData` dataclass flows through the entire pipeline
- `Config` dataclass handles validated configuration
- All methods are static for concurrent processing support

**Pipeline Flow:**
```python
data = PDFExtractor.extract(data)    # Extract text, images, tables
data = transforms.clean(data)         # Normalize text
data = transforms.chunk(data)         # Split into chunks
data = transforms.upload_to_dataloop(data)  # Upload chunks
```
