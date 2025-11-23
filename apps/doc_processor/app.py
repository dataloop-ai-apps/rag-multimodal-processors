"""
DOC/DOCX processor app.

DOCX processor that uses DOCExtractor and ExtractedData throughout.
"""

import logging
from typing import Dict, Any, List, Optional

import dtlpy as dl

import transforms
from utils.extracted_data import ExtractedData
from utils.config import Config
from .doc_extractor import DOCExtractor

logger = logging.getLogger(__name__)


class DOCProcessor(dl.BaseServiceRunner):
    """
    DOCX processing application.

    Uses ExtractedData as the data structure throughout the pipeline.
    """

    def __init__(self):
        """Initialize DOC processor."""
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Extract content from DOCX."""
        return DOCExtractor.extract(data)

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
        return DOCProcessor.run(item, target_dataset, config)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, config: Optional[Dict[str, Any]] = None) -> List[dl.Item]:
        """
        Process a DOCX document into chunks.

        Args:
            item: DOCX item to process
            target_dataset: Target dataset for storing chunks
            config: Processing configuration dict

        Returns:
            List of uploaded chunk items
        """
        cfg = Config.from_dict(config or {})
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        try:
            data = DOCProcessor.extract(data)
            data = DOCProcessor.clean(data)
            data = DOCProcessor.chunk(data)
            data = DOCProcessor.upload(data)

            logger.info(f"Processed {item.name}: {len(data.uploaded_items)} chunks, {data.errors.get_summary()}")
            return data.uploaded_items

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
