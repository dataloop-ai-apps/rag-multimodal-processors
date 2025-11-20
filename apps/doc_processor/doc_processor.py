"""
DOC/DOCX processor app.

DOCX processor with all extraction and processing logic.
"""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional

import dtlpy as dl
from docx import Document

import transforms
from utils.chunk_metadata import ChunkMetadata
from utils.data_types import ExtractedContent, ImageContent, TableContent
from utils.dataloop_helpers import upload_chunks

logger = logging.getLogger("rag-preprocessor")


class DOCProcessor(dl.BaseServiceRunner):
    """
    Unified DOCX processing application.

    Supports:
    - Text extraction from DOC/DOCX files
    - Table extraction
    - Multiple chunking strategies
    - Text cleaning and normalization
    """

    def __init__(
        self,
        item: Optional[dl.Item] = None,
        target_dataset: Optional[dl.Dataset] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DOC processor.

        Args:
            item: Dataloop DOCX item to process (optional, can be set via process_document)
            target_dataset: Target dataset for output chunks (optional, can be set via process_document)
            config: Processing configuration dict (optional, can come from context in pipeline mode)
        """
        # Configure Dataloop client timeouts
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}
        logger.info("DOCProcessor initialized")

    @staticmethod
    def extract_docx(item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """
        Extract content from DOCX (formerly in DocsExtractor).

        Args:
            item: Dataloop DOCX item to extract from
            config: Configuration dict

        Returns:
            ExtractedContent: Extracted content with text, images, and tables
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)
            doc = Document(file_path)

            result = ExtractedContent()

            # Extract paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            result.text = '\n\n'.join(paragraphs)

            # Extract images if requested
            if config.get('extract_images', True):
                result.images = DOCProcessor._extract_images(doc, temp_dir)

            # Extract tables if requested
            if config.get('extract_tables', False):
                result.tables = DOCProcessor._extract_tables(doc)

            result.metadata = {
                'paragraph_count': len(paragraphs),
                'source_file': item.name,
                'image_count': len(result.images),
                'table_count': len(result.tables),
                'processor': 'doc',
            }

            return result

    @staticmethod
    def _extract_images(doc, temp_dir: str) -> List[ImageContent]:
        """Extract embedded images from .docx"""
        images = []

        try:
            # .docx images are in doc.part.rels
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_path = os.path.join(temp_dir, rel.target_ref.split('/')[-1])
                    with open(image_path, 'wb') as f:
                        f.write(rel.target_part.blob)

                    # Extract file extension
                    ext = rel.target_ref.split('.')[-1] if '.' in rel.target_ref else None

                    images.append(
                        ImageContent(path=image_path, format=ext, caption=None, page_number=None, bbox=None, size=None)
                    )
        except Exception as e:
            logger.warning(f"Failed to extract images from .docx: {e}")

        return images

    @staticmethod
    def _extract_tables(doc) -> List[TableContent]:
        """Extract tables from .docx"""
        tables = []

        for table in doc.tables:
            try:
                # Convert to list of dicts
                rows = []
                headers = [cell.text for cell in table.rows[0].cells]

                for row in table.rows[1:]:
                    row_data = {headers[i]: cell.text for i, cell in enumerate(row.cells)}
                    rows.append(row_data)

                # Convert to markdown
                markdown = DOCProcessor._table_to_markdown(headers, rows)

                tables.append(TableContent(data=rows, markdown=markdown))
            except Exception as e:
                logger.warning(f"Failed to extract table: {e}")

        return tables

    @staticmethod
    def _table_to_markdown(headers: List[str], rows: List[Dict]) -> str:
        """Convert table to markdown format"""
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in rows:
            md += "| " + " | ".join([str(row.get(h, '')) for h in headers]) + " |\n"

        return md

    @staticmethod
    def extract(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from DOCX file."""
        item = data.get('item')
        if not item:
            raise ValueError("Missing 'item' in data")

        logger.info(f"Extracting content from: {item.name}")

        extracted = DOCProcessor.extract_docx(item, config)

        logger.info(
            f"Extracted {len(extracted.text)} chars, "
            f"{len(extracted.images)} images, "
            f"{len(extracted.tables)} tables"
        )

        # Merge extracted content into data
        data.update(extracted.to_dict())
        return data

    @staticmethod
    def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        logger.info("Cleaning text")
        data = transforms.clean_text(data, config)
        data = transforms.normalize_whitespace(data, config)
        return data

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        strategy = config.get('chunking_strategy', 'recursive')
        link_images = config.get('link_images_to_chunks', True)
        embed_images = config.get('embed_images_in_chunks', False)
        has_images = len(data.get('images', [])) > 0

        logger.info(f"Chunking with strategy: {strategy}")

        # Use image-aware chunking if images are present and strategy is recursive
        if strategy == 'recursive' and has_images:
            if embed_images:
                logger.info("Using embedded image chunking")
                data = transforms.chunk_with_embedded_images(data, config)
            elif link_images:
                logger.info("Using image-linked chunking")
                data = transforms.chunk_recursive_with_images(data, config)
            else:
                # Standard recursive chunking without image association
                data = transforms.chunk_text(data, config)
        elif strategy == 'semantic':
            # Semantic chunking uses LLM
            data = transforms.llm_chunk_semantic(data, config)
        else:
            # Use unified chunk_text for all other strategies
            data = transforms.chunk_text(data, config)

        chunk_count = len(data.get('chunks', []))
        embedded_count = sum(1 for m in data.get('chunk_metadata', []) if m.get('has_embedded_images', False))
        if embedded_count > 0:
            logger.info(f"Created {chunk_count} chunks ({embedded_count} with embedded images)")
        else:
            logger.info(f"Created {chunk_count} chunks")
        return data

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks to Dataloop with image associations."""
        logger.info("Uploading chunks to Dataloop")

        # If we have chunk metadata with image associations, use enhanced upload
        chunk_metadata = data.get('chunk_metadata', [])
        image_id_map = data.get('image_id_map', {})

        if chunk_metadata and image_id_map:
            # Update chunk metadata with actual image IDs
            for chunk_meta in chunk_metadata:
                image_indices = chunk_meta.get('image_indices', [])
                actual_image_ids = []
                for img_idx in image_indices:
                    if img_idx in image_id_map:
                        actual_image_ids.append(image_id_map[img_idx])
                chunk_meta['image_ids'] = actual_image_ids

            # Use enhanced upload that supports per-chunk metadata
            data = DOCProcessor._upload_chunks_with_metadata(data, config)
        else:
            # Standard upload
            data = transforms.upload_to_dataloop(data, config)

        uploaded_count = len(data.get('uploaded_items', []))
        logger.info(f"Uploaded {uploaded_count} chunks")
        return data

    @staticmethod
    def _upload_chunks_with_metadata(data: Dict[str, Any], _config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks with per-chunk metadata including image associations."""
        chunks = data.get('chunks', [])
        chunk_metadata_list = data.get('chunk_metadata', [])
        item = data.get('item')
        target_dataset = data.get('target_dataset')
        processor_metadata = data.get('metadata', {})

        if not chunks or not item or not target_dataset:
            data['uploaded_items'] = []
            return data

        # Create metadata for each chunk using new dataclass
        chunk_metadatas = []
        for idx, _ in enumerate(chunks):
            chunk_meta = next((m for m in chunk_metadata_list if m.get('chunk_index') == idx), {})

            # Preserve all chunk context metadata
            chunk_context = {
                **processor_metadata,  # Start with processor metadata
                **{
                    k: v
                    for k, v in chunk_meta.items()
                    if k not in ['chunk_index', 'page_numbers', 'image_ids', 'image_indices']
                },  # Add chunk-specific context
            }

            metadata = ChunkMetadata.create(
                source_item=item,
                total_chunks=len(chunks),
                chunk_index=idx,
                page_numbers=chunk_meta.get('page_numbers'),
                image_ids=chunk_meta.get('image_ids', []),
                processor='doc',
                extraction_method=processor_metadata.get('extraction_method'),
                processor_specific_metadata=chunk_context,  # Include all chunk context
            ).to_dict()  # Convert to dict for upload
            chunk_metadatas.append(metadata)

        # Upload chunks with individual metadata in a single bulk operation
        uploaded_items = upload_chunks(
            chunks=chunks,
            source_item=item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=processor_metadata,
            chunk_metadata_list=chunk_metadatas,  # Pass per-chunk metadata
        )

        data['uploaded_items'] = uploaded_items
        data['metadata']['uploaded_count'] = len(uploaded_items)

        return data

    def process_document(self, item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """
        Dataloop pipeline entry point.

        Called automatically by Dataloop pipeline nodes.

        Args:
            item: DOCX item to process
            target_dataset: Target dataset for storing chunks
            context: Processing context with configuration

        Returns:
            List of uploaded chunk items
        """
        # Set item and dataset for pipeline mode
        self.item = item
        self.target_dataset = target_dataset if target_dataset is not None else item.dataset

        # Get configuration from node
        node = context.node
        self.config = node.metadata.get('customNodeConfig', {})

        # Execute the processing pipeline
        return self.run()

    def run(self) -> List[dl.Item]:
        """
        Execute the full processing pipeline.

        Returns:
            List of uploaded chunk items
        """
        if self.item is None or self.target_dataset is None:
            raise ValueError("Item and target_dataset must be set before calling run()")

        logger.info(f"Starting DOC processing: {self.item.name}")

        try:
            # Initialize data with context
            data = {'item': self.item, 'target_dataset': self.target_dataset}

            # Execute pipeline stages sequentially using static methods
            data = DOCProcessor.extract(data, self.config)
            data = DOCProcessor.clean(data, self.config)
            data = DOCProcessor.chunk(data, self.config)
            data = DOCProcessor.upload(data, self.config)

            # Return uploaded items
            uploaded = data.get('uploaded_items', [])

            logger.info(f"Processing complete: {len(uploaded)} chunks")
            return uploaded

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
