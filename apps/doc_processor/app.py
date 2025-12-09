"""
DOC/DOCX processor app.

DOCX processor that uses DOCExtractor and ExtractedData throughout.
"""

import logging
from typing import List

import dtlpy as dl
import nltk

import transforms
from utils.extracted_data import ExtractedData
from utils.config import Config
from apps.doc_processor.doc_extractor import DOCExtractor

logger = logging.getLogger("rag-preprocessor")


class DOCProcessor(dl.BaseServiceRunner):
    """DOCX processing application."""

    def __init__(self):
        """Initialize DOC processor."""
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

        for resource in ['tokenizers/punkt', 'taggers/averaged_perceptron_tagger']:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split('/')[-1], quiet=True)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Process a DOCX document into chunks."""
        config = context.node.metadata.get('customNodeConfig', {})
        cfg = Config.from_dict(config)
        data = ExtractedData(item=item, target_dataset=target_dataset, config=cfg)

        try:
            data = DOCExtractor.extract(data)
            if cfg.ocr_from_images:
                data = transforms.ocr_enhance(data)
            if cfg.to_correct_spelling:
                data = transforms.deep_clean(data)
            else:
                data = transforms.clean(data)
            data = transforms.chunk(data)
            data = transforms.upload_to_dataloop(data)

            logger.info(f"Processed {item.name}: {len(data.uploaded_items)} chunks, {data.errors.get_summary()}")
            return data.uploaded_items

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
