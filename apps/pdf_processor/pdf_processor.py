"""
PDF processor app.

Uses existing extractors and operations from the repo for processing PDFs.
"""

import logging
from typing import Dict, Any, List, Optional
import dtlpy as dl

from extractors import PDFExtractor
import operations
from utils.dataloop_helpers import upload_chunks
from utils.chunk_metadata import ChunkMetadata

logger = logging.getLogger("rag-preprocessor")


class PDFProcessor(dl.BaseServiceRunner):
    """
    Unified PDF Processor for extracting text, applying OCR, and creating chunks.

    Supports:
    - Text extraction (plain and markdown-aware)
    - Image extraction and OCR
    - Multiple chunking strategies
    - Text cleaning and normalization
    """

    def __init__(self, item: Optional[dl.Item] = None, target_dataset: Optional[dl.Dataset] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF processor.

        Args:
            item: Dataloop PDF item to process (optional, can be set via process_document)
            target_dataset: Target dataset for output chunks (optional, can be set via process_document)
            config: Processing configuration dict (optional, can come from context in pipeline mode)
        """
        # Configure Dataloop client timeouts
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

        self.item = item
        self.target_dataset = target_dataset
        self.config = config

        # Initialize extractor (doesn't depend on item/dataset/config)
        self.extractor = PDFExtractor()
        logger.info("PDFProcessor initialized")

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from PDF file."""
        logger.info(f"Extracting content from: {self.item.name}")

        extracted = self.extractor.extract(self.item, self.config)

        logger.info(
            f"Extracted {len(extracted.text)} chars, "
            f"{len(extracted.images)} images, "
            f"{len(extracted.tables)} tables"
        )

        # Merge extracted content into data
        data.update(extracted.to_dict())
        return data

    def apply_ocr(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OCR if enabled in config."""
        # Support both 'ocr_from_images' (from dataloop.json) and 'use_ocr' (legacy)
        ocr_enabled = self.config.get('ocr_from_images', False) or self.config.get('use_ocr', False)
        if not ocr_enabled:
            logger.debug("OCR disabled, skipping")
        else:
            logger.info("Applying OCR to images")
            # Map dataloop.json config values to ocr_enhance function values
            ocr_config = self.config.copy()
            ocr_config['use_ocr'] = True
            
            # Map integration method values
            integration_method = ocr_config.get('ocr_integration_method', 'append_to_page')
            method_mapping = {
                'append_to_page': 'per_page',  # Map dataloop.json value to ocr_enhance value
                'separate_chunks': 'separate',
                'combine_all': 'append',
            }
            ocr_config['ocr_integration_method'] = method_mapping.get(integration_method, integration_method)
            
            data = operations.ocr_enhance(data, ocr_config)
        return data

    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        logger.info("Cleaning text")
        data = operations.clean_text(data, self.config)
        data = operations.normalize_whitespace(data, self.config)
        return data

    def chunk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        strategy = self.config.get('chunking_strategy', 'recursive')
        link_images = self.config.get('link_images_to_chunks', True)
        embed_images = self.config.get('embed_images_in_chunks', False)
        has_images = len(data.get('images', [])) > 0

        logger.info(f"Chunking with strategy: {strategy}")

        # Use image-aware chunking if images are present and strategy is recursive
        if strategy == 'recursive' and has_images:
            if embed_images:
                logger.info("Using embedded image chunking")
                data = operations.chunk_with_embedded_images(data, self.config)
            elif link_images:
                logger.info("Using image-linked chunking")
                data = operations.chunk_recursive_with_images(data, self.config)
            else:
                # Standard recursive chunking without image association
                data = operations.chunk_text(data, self.config)
        elif strategy == 'semantic':
            # Semantic chunking uses LLM
            data = operations.llm_chunk_semantic(data, self.config)
        else:
            # Use unified chunk_text for all other strategies
            data = operations.chunk_text(data, self.config)

        chunk_count = len(data.get('chunks', []))
        embedded_count = sum(1 for m in data.get('chunk_metadata', []) if m.get('has_embedded_images', False))
        if embedded_count > 0:
            logger.info(f"Created {chunk_count} chunks ({embedded_count} with embedded images)")
        else:
            logger.info(f"Created {chunk_count} chunks")
        return data

    def upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
            data = self._upload_chunks_with_metadata(data, self.config)
        else:
            # Standard upload
            data = operations.upload_to_dataloop(data, self.config)

        uploaded_count = len(data.get('uploaded_items', []))
        logger.info(f"Uploaded {uploaded_count} chunks")
        return data

    def _upload_chunks_with_metadata(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks with per-chunk metadata including image associations."""
        chunks = data.get('chunks', [])
        chunk_metadata_list = data.get('chunk_metadata', [])
        item = data.get('item')
        target_dataset = data.get('target_dataset')
        processor_metadata = data.get('metadata', {})

        if not chunks or not item or not target_dataset:
            data['uploaded_items'] = []
            return data

        # Create metadata for each chunk
        chunk_metadatas = []
        for idx, _ in enumerate(chunks):
            chunk_meta = next((m for m in chunk_metadata_list if m.get('chunk_index') == idx), {})

            metadata = ChunkMetadata.create(
                original_item=item,
                total_chunks=len(chunks),
                processor_specific_metadata=processor_metadata,
                chunk_index=idx,
                page_numbers=chunk_meta.get('page_numbers'),
                image_ids=chunk_meta.get('image_ids', []),
            )
            chunk_metadatas.append(metadata)

        # Upload chunks with individual metadata
        # Note: upload_chunks currently uses same metadata for all chunks
        # We'll need to upload individually or enhance upload_chunks
        # For now, upload with first chunk's metadata structure
        uploaded_items = upload_chunks(
            chunks=chunks,
            original_item=item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=processor_metadata,
        )

        # Update each uploaded item with its specific metadata
        for idx, uploaded_item in enumerate(uploaded_items):
            if idx < len(chunk_metadatas):
                try:
                    uploaded_item.metadata = chunk_metadatas[idx]
                    uploaded_item.update(system_metadata=True)
                except Exception as e:
                    logger.warning(f"Failed to update metadata for chunk {idx}: {e}")

        data['uploaded_items'] = uploaded_items
        data['metadata']['uploaded_count'] = len(uploaded_items)

        return data

    def process_document(self, item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """
        Dataloop pipeline entry point.

        Called automatically by Dataloop pipeline nodes.

        Args:
            item: PDF item to process
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

        TODO: Add tests for:
            - Image extraction
            - Image-chunk association
            - Config options (extract_images, link_images_to_chunks)
            - PDFs with and without images
        """
        if self.item is None or self.target_dataset is None:
            raise ValueError("Item and target_dataset must be set before calling run()")

        logger.info(f"Starting PDF processing: {self.item.name}")

        try:
            # Initialize data with context
            data = {'item': self.item, 'target_dataset': self.target_dataset}

            # Execute pipeline stages sequentially
            data = self.extract(data)
            data = self.apply_ocr(data)
            data = self.clean(data)

            data = self.chunk(data)
            data = self.upload(data)

            # Return uploaded items
            uploaded = data.get('uploaded_items', [])

            logger.info(f"Processing complete: {len(uploaded)} chunks")
            return uploaded

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
