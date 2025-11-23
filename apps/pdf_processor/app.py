"""
PDF processor app.

PDF processor that uses PDFExtractor and ExtractedData throughout.
"""

import logging
from typing import Dict, Any, List, Optional

import dtlpy as dl
import nltk

import transforms
from utils.extracted_data import ExtractedData
from utils.config import Config
from .pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class PDFProcessor(dl.BaseServiceRunner):
    """
    PDF Processor for extracting text, applying OCR, and creating chunks.

    Uses ExtractedData as the data structure throughout the pipeline.
    """

    def __init__(self):
        """Initialize PDF processor."""
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

        for resource in ['tokenizers/punkt', 'taggers/averaged_perceptron_tagger']:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split('/')[-1], quiet=True)

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Extract content from PDF."""
        return PDFExtractor.extract(data)

    @staticmethod
    def apply_ocr(data: ExtractedData) -> ExtractedData:
        """Apply OCR if enabled."""
        if not data.config.use_ocr:
            return data
        return transforms.ocr_enhance(data)

    @staticmethod
    def clean(data: ExtractedData) -> ExtractedData:
        """Clean and normalize text."""
        return transforms.clean(data)

    @staticmethod
    def chunk(data: ExtractedData) -> ExtractedData:
        """Chunk content based on strategy."""
        strategy = data.config.chunking_strategy
        has_images = data.has_images()

        if strategy == 'recursive' and has_images:
            return transforms.chunk_with_images(data)
        elif strategy == 'semantic':
            return transforms.llm_chunk_semantic(data)
        else:
            return transforms.chunk(data)

    @staticmethod
    def upload(data: ExtractedData) -> ExtractedData:
        """Upload chunks to Dataloop."""
        return transforms.upload_to_dataloop(data)

    @staticmethod
    def process_document(item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Dataloop pipeline entry point."""
        config = context.node.metadata.get('customNodeConfig', {})
        return PDFProcessor.run(item, target_dataset, config)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, config: Optional[Dict[str, Any]] = None) -> List[dl.Item]:
        """
        Process a PDF document into chunks.

        Args:
            item: PDF item to process
            target_dataset: Target dataset for storing chunks
            config: Processing configuration dict

        Returns:
            List of uploaded chunk items
        """
        cfg = Config.from_dict(config or {})
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        try:
            data = PDFProcessor.extract(data)
            data = PDFProcessor.apply_ocr(data)
            data = PDFProcessor.clean(data)
            data = PDFProcessor.chunk(data)
            data = PDFProcessor.upload(data)

            logger.info(f"Processed {item.name}: {len(data.uploaded_items)} chunks, {data.errors.get_summary()}")
            return data.uploaded_items

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
