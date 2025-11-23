# DOC Processor

A modular Dataloop application for processing DOCX files into RAG-ready chunks with support for text, tables, and images.

## 🎯 Features

### Text Extraction
- **Paragraph Extraction**: Extracts all text paragraphs from DOCX files using `python-docx`
- **Table Extraction**: Optional extraction of tables as markdown format
- **Image Extraction**: Optional extraction of embedded images

### Chunk Upload
- **Bulk Upload**: Upload all chunks in a single operation using pandas DataFrame
- **Per-Chunk Metadata**: Track images and extraction details per chunk
- **Resilient Fallback**: Automatic retry with individual uploads if needed

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
| `extract_tables` | boolean | `false` | `true` or `false` | When enabled, extracts all tables from the DOCX document and includes them as markdown-formatted text in the chunks. Tables are converted to markdown format with headers and rows. |
| `extract_images` | boolean | `true` | `true` or `false` | When enabled, extracts embedded images from the DOCX document. Images are included in the extraction metadata but not directly in chunk text. |
| `chunking_strategy` | string | `"recursive"` | `"recursive"`, `"fixed-size"`, `"nltk-sentence"`, `"nltk-paragraphs"`, or `"1-chunk"` | Determines how the extracted text is split into chunks. `recursive`: Intelligently splits text respecting semantic boundaries (recommended for most use cases). `fixed-size`: Creates uniform chunks of equal size with overlap. `nltk-sentence`: Splits by sentences using NLTK. `nltk-paragraphs`: Splits by paragraphs using NLTK. `1-chunk`: No splitting, entire document as one chunk (useful for short documents). |
| `max_chunk_size` | integer | `300` | `1` to `2000` characters | Maximum size of each text chunk in characters. Smaller chunks provide more granular retrieval but may lose context. Larger chunks maintain more context but may be less precise. Recommended range: 300-500 for most RAG applications. This parameter works with `chunk_overlap` to control chunk boundaries. |
| `chunk_overlap` | integer | `20` | `0` to `400` characters | Number of characters that overlap between consecutive chunks. Overlap helps maintain context across chunk boundaries and prevents information loss at split points. Should be 10-20% of `max_chunk_size`. Set to 0 for no overlap. Higher values improve context preservation but increase storage requirements. |
| `correct_spelling` | boolean | `false` | `true` or `false` | When enabled, applies text cleaning and normalization using the `unstructured.io` library. This includes: unicode quote replacement, non-ASCII character handling, punctuation normalization, and whitespace normalization. Enable for documents with inconsistent formatting. May slow down processing. |

## 🚀 Deployment

1. Go to Dataloop Marketplace
2. Search for "DOC to Chunks"
3. Click "Install"
4. Configure in your pipeline

## 📊 Usage

### In a Pipeline

1. **Add the app node** to your pipeline
2. **Configure parameters** in the node settings
3. **Connect input** (DOCX items from dataset)
4. **Connect output** (chunks will be created in target dataset)
5. **Run the pipeline**

### Example Pipeline Flow

```
[DOCX Dataset] → [DOC to Chunks] → [Embedding Model] → [Vector Database]
```

### Example Configuration for Production

```json
{
  "extract_tables": true,
  "extract_images": true,
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
document.docx → document_chunk_0001.txt
              → document_chunk_0002.txt
              → document_chunk_0003.txt
```

### Metadata Structure

Each chunk includes comprehensive metadata for provenance tracking:

```json
{
  "user": {
    "source_item_id": "65f2a3b4c1e2d3f4a5b6c7d8",
    "source_file": "example.docx",
    "source_dataset_id": "65f2a3b4c1e2d3f4a5b6c7d9",
    "chunk_index": 0,
    "total_chunks": 45,
    "extracted_chunk": true,
    "processing_timestamp": 1698765432.123,
    "processor": "doc",
    "extraction_method": "python-docx",
    "image_ids": ["img_id_1", "img_id_2", "img_id_3", "img_id_4", "img_id_5"]
  }
}
```

## 🏗️ Architecture

The DOC processor uses a type-safe, stateless architecture:

```
DOCProcessor (app.py)
    ├── DOCExtractor (doc_extractor.py) - DOCX-specific extraction
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
data = DOCExtractor.extract(data)    # Extract text, images, tables
data = transforms.clean(data)         # Normalize text
data = transforms.chunk(data)         # Split into chunks
data = transforms.upload_to_dataloop(data)  # Upload chunks
```

