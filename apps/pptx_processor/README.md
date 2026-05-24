# PPTX Processor

A modular Dataloop application for processing PowerPoint files into RAG-ready chunks with slide text, speaker notes, tables, and image extraction.

## 🎯 Features

### Text Extraction
- **Slide Text**: Extracts all text from titles, body text boxes, and placeholders per slide
- **Speaker Notes**: Extracts presenter notes attached to each slide
- **Tables**: Extracts table content as formatted text rows

### Image Extraction
- Extracts embedded images from slides with slide number and positional metadata
- Stores images as temporary files for downstream processing

### Chunk Upload
- **Bulk Upload**: Upload all chunks in a single operation
- **Per-Chunk Metadata**: Track slide numbers, extraction method per chunk
- **Resilient Fallback**: Automatic retry with individual uploads if needed

### Text Chunking
Multiple strategies for optimal embedding:
- `recursive`: Intelligent splitting respecting semantic boundaries
- `fixed-size`: Uniform chunks with configurable overlap
- `nltk-sentence`: Sentence-based chunking
- `nltk-paragraphs`: Paragraph-based chunking
- `1-chunk`: Single chunk (for short presentations)

### Text Processing
- **Text Cleaning**: Optional normalization using `unstructured.io`

## ⚙️ Configuration

All parameters are configured via the pipeline node in Dataloop.

### Core Parameters

| Parameter | Type | Default | Valid Values/Range | Description |
|-----------|------|---------|-------------------|-------------|
| `extract_images` | boolean | `true` | `true` or `false` | Extract embedded images from slides |
| `extract_tables` | boolean | `true` | `true` or `false` | Extract tables from slides as formatted text |
| `to_correct_spelling` | boolean | `false` | `true` or `false` | Apply text cleaning and normalization |
| `chunking_strategy` | string | `"recursive"` | `"recursive"`, `"fixed-size"`, `"nltk-sentence"`, `"nltk-paragraphs"`, `"1-chunk"` | Method for splitting text into chunks |
| `max_chunk_size` | integer | `300` | `1` to `2000` characters | Maximum size of each text chunk in characters |
| `chunk_overlap` | integer | `40` | `0` to `400` characters | Number of overlapping characters between consecutive chunks |

### Example Configuration

```json
{
  "extract_images": true,
  "extract_tables": true,
  "chunking_strategy": "recursive",
  "max_chunk_size": 500,
  "chunk_overlap": 50,
  "to_correct_spelling": false
}
```

## 🚀 Deployment

1. Go to Dataloop Marketplace
2. Search for "PPTX to Chunks"
3. Click "Install"
4. Configure in your pipeline

## 📊 Usage

### In a Pipeline

1. **Add the app node** to your pipeline
2. **Configure parameters** in the node settings
3. **Connect input** (PPTX items from dataset)
4. **Connect output** (chunks will be created in target dataset)
5. **Run the pipeline**

### Example Pipeline Flow

```
[PPTX Dataset] → [PPTX to Chunks] → [Embedding Model] → [Vector Database]
```

## 📤 Output Format

### Chunk Items

Each chunk is uploaded as a text file (`.txt`):
```
{original_filename}_chunk_{index:04d}.txt
```

Example:
```
presentation.pptx → presentation_chunk_0001.txt
                  → presentation_chunk_0002.txt
                  → presentation_chunk_0003.txt
```

### Metadata Structure

Each chunk includes metadata for provenance tracking:

```json
{
  "user": {
    "source_item_id": "65f2a3b4c1e2d3f4a5b6c7d8",
    "source_file": "presentation.pptx",
    "source_dataset_id": "65f2a3b4c1e2d3f4a5b6c7d9",
    "chunk_index": 0,
    "total_chunks": 20,
    "extracted_chunk": true,
    "processing_timestamp": 1698765432.123,
    "processor": "pptx",
    "extraction_method": "python-pptx",
    "slide_count": 15,
    "image_count": 5,
    "table_count": 2
  }
}
```

## 🏗️ Architecture

```
PPTXProcessor (app.py)
    ├── PPTXExtractor (pptx_extractor.py) - PPTX-specific extraction
    └── Transforms - Shared pipeline operations
        ├── transforms.clean() - Text normalization
        ├── transforms.chunk() - Text chunking
        └── transforms.upload_to_dataloop() - Chunk upload
```

**Pipeline Flow:**
```python
data = PPTXExtractor.extract(data)       # Extract text, notes, tables, images
data = transforms.clean(data)             # Normalize text
data = transforms.chunk(data)             # Split into chunks
data = transforms.upload_to_dataloop(data)  # Upload chunks
```
