"""
DOC/DOCX processor app.

DOCX processor that uses DOCExtractor and ExtractedData throughout.
"""

import logging
from typing import List

import dtlpy as dl

import transforms
from utils.extracted_data import ExtractedData
from utils.config import Config
from .doc_extractor import DOCExtractor

logger = logging.getLogger(__name__)


class DOCProcessor(dl.BaseServiceRunner):
    """DOCX processing application."""

    def __init__(self):
        """Initialize DOC processor."""
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Process a DOCX document into chunks."""
        config = context.node.metadata.get('customNodeConfig', {})
        cfg = Config.from_dict(config)
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        try:
            data = DOCExtractor.extract(data)
            data = transforms.clean(data)
            data = transforms.chunk(data)
            data = transforms.upload_to_dataloop(data)

            logger.info(f"Processed {item.name}: {len(data.uploaded_items)} chunks, {data.errors.get_summary()}")
            return data.uploaded_items

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
