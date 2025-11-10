"""
Simple PDF processor app.

Uses existing extractors and stages from the repo for processing PDFs.
No complex inheritance - just straightforward function composition.
"""

import logging
from typing import Dict, Any, List
import dtlpy as dl

# Import existing utilities from repo
from extractors import PDFExtractor
import stages


class PDFApp:
    """
    Simple PDF processing application.

    Usage:
        >>> app = PDFApp(
        ...     item=pdf_item,
        ...     target_dataset=chunks_dataset,
        ...     config={'use_ocr': True, 'max_chunk_size': 500}
        ... )
        >>> chunks = app.run()
    """

    def __init__(
        self,
        item: dl.Item,
        target_dataset: dl.Dataset,
        config: Dict[str, Any] = None
    ):
        """
        Initialize PDF processor.

        Args:
            item: Dataloop PDF item to process
            target_dataset: Target dataset for output chunks
            config: Processing configuration dict
        """
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}
        self.extractor = PDFExtractor()

        # Setup logging
        log_level = self.config.get('log_level', 'INFO')
        self.logger = logging.getLogger(f"PDFApp.{item.id[:8]}")
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from PDF file."""
        self.logger.info(f"Extracting content from: {self.item.name}")

        extracted = self.extractor.extract(self.item, self.config)

        self.logger.info(
            f"Extracted {len(extracted.text)} chars, "
            f"{len(extracted.images)} images, "
            f"{len(extracted.tables)} tables"
        )

        # Merge extracted content into data
        data.update(extracted.to_dict())
        return data

    def apply_ocr(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OCR if enabled in config."""
        if not self.config.get('use_ocr', False):
            self.logger.debug("OCR disabled, skipping")
            return data

        self.logger.info("Applying OCR to images")
        data = stages.ocr_enhance(data, self.config)
        return data

    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        self.logger.info("Cleaning text")
        data = stages.clean_text(data, self.config)
        data = stages.normalize_whitespace(data, self.config)
        return data

    def upload_images(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload extracted images to Dataloop before temp cleanup."""
        images = data.get('images', [])
        extract_images = self.config.get('extract_images', True)
        upload_images = self.config.get('upload_images', True)

        if not extract_images or not images:
            self.logger.debug("No images to upload")
            return data

        if not upload_images:
            self.logger.debug("Image upload disabled in config")
            return data

        self.logger.info(f"Uploading {len(images)} images to Dataloop")
        data = stages.upload_with_images(data, self.config)

        uploaded_count = len(data.get('uploaded_images', []))
        self.logger.info(f"Uploaded {uploaded_count} images")

        # Create mapping from image index to uploaded item ID
        uploaded_images = data.get('uploaded_images', [])
        image_id_map = {}  # Maps image index -> uploaded item ID
        for idx, uploaded_img in enumerate(uploaded_images):
            if hasattr(uploaded_img, 'id'):
                image_id_map[idx] = uploaded_img.id
            else:
                image_id_map[idx] = str(uploaded_img)

        data['image_id_map'] = image_id_map
        return data

    def chunk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        strategy = self.config.get('chunking_strategy', 'recursive')
        link_images = self.config.get('link_images_to_chunks', True)
        embed_images = self.config.get('embed_images_in_chunks', False)
        has_images = len(data.get('images', [])) > 0

        self.logger.info(f"Chunking with strategy: {strategy}")

        # Use image-aware chunking if images are present
        if strategy == 'recursive' and has_images:
            if embed_images:
                # Embed images directly in chunk text
                self.logger.info("Using embedded image chunking")
                data = stages.chunk_recursive_with_images(data, self.config)
            elif link_images:
                # Associate images with chunks via metadata
                self.logger.info("Using image-linked chunking")
                data = stages.chunk_recursive_with_images(data, self.config)
            else:
                # Standard chunking without image association
                data = stages.chunk_recursive(data, self.config)
        elif strategy == 'recursive':
            data = stages.chunk_recursive(data, self.config)
        elif strategy == 'semantic':
            data = stages.llm_chunk_semantic(data, self.config)
        elif strategy == 'sentence':
            data = stages.chunk_by_sentence(data, self.config)
        elif strategy == 'paragraph':
            data = stages.chunk_by_paragraph(data, self.config)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        chunk_count = len(data.get('chunks', []))
        embedded_count = sum(1 for m in data.get('chunk_metadata', []) if m.get('has_embedded_images', False))
        if embedded_count > 0:
            self.logger.info(f"Created {chunk_count} chunks ({embedded_count} with embedded images)")
        else:
            self.logger.info(f"Created {chunk_count} chunks")
        return data

    def upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks to Dataloop with image associations."""
        self.logger.info("Uploading chunks to Dataloop")

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
            data = stages.upload_to_dataloop(data, self.config)

        uploaded_count = len(data.get('uploaded_items', []))
        self.logger.info(f"Uploaded {uploaded_count} chunks")
        return data

    def _upload_chunks_with_metadata(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks with per-chunk metadata including image associations."""
        try:
            from utils.dataloop_helpers import upload_chunks
            from utils.chunk_metadata import ChunkMetadata
        except ImportError:
            self.logger.warning("dataloop_helpers not found, using standard upload")
            return stages.upload_to_dataloop(data, config)

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
        for idx, chunk in enumerate(chunks):
            chunk_meta = next(
                (m for m in chunk_metadata_list if m.get('chunk_index') == idx),
                {}
            )

            metadata = ChunkMetadata.create(
                original_item=item,
                total_chunks=len(chunks),
                processor_specific_metadata=processor_metadata,
                chunk_index=idx,
                page_numbers=chunk_meta.get('page_numbers'),
                image_ids=chunk_meta.get('image_ids', [])
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
            processor_metadata=processor_metadata
        )

        # Update each uploaded item with its specific metadata
        for idx, uploaded_item in enumerate(uploaded_items):
            if idx < len(chunk_metadatas):
                try:
                    uploaded_item.metadata = chunk_metadatas[idx]
                    uploaded_item.update(system_metadata=True)
                except Exception as e:
                    self.logger.warning(f"Failed to update metadata for chunk {idx}: {e}")

        data['uploaded_items'] = uploaded_items
        data['metadata']['uploaded_count'] = len(uploaded_items)

        return data

    def run(self) -> List[dl.Item]:
        """
        Execute the full processing pipeline.

        Returns:
            List of uploaded chunk items

        TODO: Add tests for:
            - Image extraction and upload
            - Image-chunk association
            - Config options (extract_images, upload_images, link_images_to_chunks)
            - PDFs with and without images
        """
        self.logger.info(f"Starting PDF processing: {self.item.name}")

        try:
            # Initialize data with context
            data = {
                'item': self.item,
                'target_dataset': self.target_dataset
            }

            # Execute pipeline stages sequentially
            data = self.extract(data)
            data = self.apply_ocr(data)
            data = self.clean(data)
            
            # Upload images BEFORE chunking (so we have image IDs for chunk metadata)
            # This must happen before temp directory cleanup
            data = self.upload_images(data)
            
            data = self.chunk(data)
            data = self.upload(data)

            # Return uploaded items
            uploaded = data.get('uploaded_items', [])
            uploaded_images = data.get('uploaded_images', [])
            
            self.logger.info(
                f"Processing complete: {len(uploaded)} chunks, "
                f"{len(uploaded_images)} images created"
            )
            return uploaded

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
