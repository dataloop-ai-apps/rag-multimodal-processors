"""
OCR extractor for extracting text from images.
Supports Dataloop models/services and external OCR libraries.
"""

import os
from pathlib import Path
import dtlpy as dl
import logging
import tempfile
import easyocr
from typing import Optional, List, Dict, Any
from .mixins import DataloopModelMixin
from .data_types import ExtractedContent

logger = logging.getLogger("rag-preprocessor")


class OCRExtractor(DataloopModelMixin):
    """
    Extract text from images using OCR.
    Defaults to EasyOCR, supports custom Dataloop models via custom_ocr_model_id.
    """

    _easyocr_reader = None
    _easyocr_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize OCR extractor.

        Args:
            model_id (Optional[str]): Dataloop model ID for custom OCR models.
                If None, uses EasyOCR for local processing.
        """
        super().__init__(model_id=model_id)
        self.mime_type = 'image/*'
        self.name = 'OCR'

    def extract(self, item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """
        Extract text from an image item using OCR.

        Args:
            item: Dataloop image item to process
            config: Configuration dictionary (can contain 'use_dataloop_model' flag)

        Returns:
            ExtractedContent: Object containing extracted OCR text
        """
        result = ExtractedContent()

        # Download image to temp location
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)

            # Extract text using appropriate method
            if config.get('use_dataloop_model', False) and self.has_dataloop_backend():
                # Use Dataloop model
                updated_item = self.execute_model(item)
                ocr_text = self._parse_ocr_result(updated_item)
            else:
                # Use EasyOCR
                ocr_text = self.extract_text(file_path)

            result.text = ocr_text
            result.metadata = {
                'source_file': item.name,
                'extraction_method': (
                    'dataloop_model'
                    if (config.get('use_dataloop_model', False) and self.has_dataloop_backend())
                    else 'easyocr'
                ),
                'extractor': 'ocr',
            }

        return result

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image file using EasyOCR.
        This method is for local file processing only.

        For Dataloop model processing, use extract_text_batch() instead.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Extracted text from the image

        Raises:
            ValueError: If custom model is configured (use extract_text_batch instead)
        """
        if self.has_dataloop_backend():
            raise ValueError(
                f"extract_text() is for EasyOCR only. " f"For custom Dataloop models, use extract_text_batch() instead."
            )

        logger.debug("Using EasyOCR for local image file")
        return self._external_ocr(image_path)

    def _parse_ocr_result(self, item: dl.Item) -> str:
        """
        Parse OCR result from updated Dataloop item.
        Priority: item.description first, then text annotations.

        Args:
            item (dl.Item): Updated item after model execution

        Returns:
            str: Extracted text
        """
        extracted_text = ""

        # Priority 1: Check item.description
        if item.description and item.description.strip():
            logger.info(f"Found OCR text in item.description | length={len(item.description)}")
            extracted_text = item.description.strip()
        else:
            # Priority 2: Check annotations with label='Text'
            try:
                annotations = item.annotations.list()
                text_annotations = []

                for annotation in annotations:
                    if hasattr(annotation, 'label') and annotation.label == 'Text':
                        if hasattr(annotation, 'description') and annotation.description:
                            text_annotations.append(annotation.description)
                            logger.debug(f"Found text annotation | text={annotation.description[:50]}...")

                if text_annotations:
                    extracted_text = ' '.join(text_annotations)
                    logger.info(f"Found {len(text_annotations)} text annotations | total_length={len(extracted_text)}")
                else:
                    logger.warning(f"No OCR results found in item description or annotations for item {item.id}")
            except Exception as e:
                logger.warning(f"Failed to parse annotations: {str(e)}")
                logger.warning(f"No OCR results found in item description or annotations for item {item.id}")

        return extracted_text

    def _external_ocr(self, image_path: str) -> str:
        """
        External OCR implementation using EasyOCR with model caching.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Extracted text
        """
        try:
            # Initialize EasyOCR reader only once (class-level caching)
            if OCRExtractor._easyocr_reader is None:
                logger.info(f"Initializing EasyOCR reader with languages: {OCRExtractor._easyocr_languages}")
                OCRExtractor._easyocr_reader = easyocr.Reader(OCRExtractor._easyocr_languages, gpu=False)
                logger.info("EasyOCR reader initialized and cached")
            else:
                logger.debug("Using cached EasyOCR reader")

            # Perform OCR directly on image path
            # Resolve path to avoid issues with Windows short path format
            resolved_path = str(Path(image_path).resolve())
            logger.debug(f"Running EasyOCR on image: {resolved_path}")
            results = OCRExtractor._easyocr_reader.readtext(resolved_path)

            # Extract text from results (bbox, text, confidence)
            all_text = ' '.join([text for (bbox, text, confidence) in results])

            logger.info(f"EasyOCR extracted {len(results)} text blocks, total length: {len(all_text)}")
            return all_text

        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return f"[OCR_ERROR: {str(e)}]"

    def extract_text_batch(self, items: List[dl.Item]) -> Dict[str, str]:
        """
        Run batch OCR on multiple Dataloop items.
        Only works with custom Dataloop models.

        Args:
            items (List[dl.Item]): List of Dataloop items to process

        Returns:
            Dict[str, str]: Dictionary mapping item_id to extracted OCR text

        Raises:
            ValueError: If custom model is not configured or not deployed
            Exception: If batch execution fails
        """
        if not self.has_dataloop_backend():
            raise ValueError(
                "Batch OCR requires a custom_ocr_model_id. Use extract_text() with EasyOCR for single images."
            )

        if not items:
            return {}

        item_ids = [item.id for item in items]
        logger.info(f"Running batch OCR prediction | model_id={self.model_id} item_count={len(item_ids)}")

        try:
            # Check model deployment (using mixin method)
            model, _ = self._check_model_deployed(auto_deploy=False)

            # Execute batch prediction
            execution = model.predict(item_ids=item_ids)
            logger.info(f"Waiting for batch execution to complete | execution_id={execution.id}")
            execution.wait()

            # Check execution status
            if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
                raise Exception(
                    f"Batch OCR execution failed: {execution.latest_status.get('message', 'Unknown error')}"
                )
            elif execution.latest_status['status'] == dl.ExecutionStatus.SUCCESS:
                logger.info(f"Batch OCR execution successful | execution_id={execution.id}")

            # Fetch results from each item
            results = {}
            for item in items:
                try:
                    # Refresh item to get OCR results
                    updated_item = dl.items.get(item_id=item.id)
                    ocr_text = self._parse_ocr_result(updated_item)
                    results[item.id] = ocr_text
                    logger.debug(f"OCR text retrieved | item_id={item.id} text_length={len(ocr_text)}")
                except Exception as e:
                    logger.warning(f"Failed to retrieve OCR results for item {item.id}: {e}")
                    results[item.id] = ""

            logger.info(f"Batch OCR completed | successful={len([r for r in results.values() if r])}/{len(items)}")
            return results

        except Exception as e:
            logger.error(f"Batch OCR prediction failed: {str(e)}")
            raise
