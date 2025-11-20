"""
OCR utilities for extracting text from images.
Consolidates OCR functionality with support for Dataloop models and EasyOCR fallback.

Note: EasyOCR uses torch.ao.quantization which is deprecated in PyTorch 2.10+.
The warning is suppressed here until EasyOCR is updated to use torchao.
See: https://github.com/pytorch/ao/issues/2259
"""

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import dtlpy as dl

# Suppress PyTorch quantization deprecation warning from EasyOCR
# EasyOCR hasn't updated to use torchao yet (as of v1.7.2)
warnings.filterwarnings(
    'ignore',
    category=DeprecationWarning,
    module='torch.ao.quantization',
    message='.*torch.ao.quantization.*'
)

# Lazy import to avoid loading torch if OCR is not used
_easyocr = None

def _get_easyocr():
    """Lazy import easyocr to avoid loading heavy dependencies until needed."""
    global _easyocr
    if _easyocr is None:
        import easyocr as _easyocr_module
        _easyocr = _easyocr_module
    return _easyocr

logger = logging.getLogger("rag-preprocessor")


class OCRProcessor:
    """
    Consolidated OCR processor with conditional logic.
    If model_id provided → use Dataloop model
    Otherwise → use EasyOCR as fallback
    """

    _easyocr_reader = None
    _easyocr_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']

    @staticmethod
    def extract_text(item: dl.Item, config: Dict[str, Any]) -> str:
        """
        Single OCR method with conditional logic.
        If model_id provided → use Dataloop model
        Otherwise → use EasyOCR as fallback

        Args:
            item: Dataloop image item to process
            config: Configuration dict with optional 'custom_ocr_model_id'

        Returns:
            str: Extracted OCR text
        """
        model_id = config.get('custom_ocr_model_id')

        if model_id:
            # Use Dataloop model
            return OCRProcessor._extract_with_model(item, model_id)
        else:
            # Use EasyOCR fallback
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)
                return OCRProcessor._extract_with_easyocr(file_path)

    @staticmethod
    def extract_text_from_path(image_path: str) -> str:
        """
        Extract text from local image file path using EasyOCR.

        Args:
            image_path: Path to image file

        Returns:
            str: Extracted text
        """
        return OCRProcessor._extract_with_easyocr(image_path)

    @staticmethod
    def _extract_with_model(item: dl.Item, model_id: str) -> str:
        """
        Extract text using Dataloop OCR model.

        Args:
            item: Dataloop image item
            model_id: Dataloop model ID

        Returns:
            str: Extracted OCR text
        """
        try:
            model = dl.models.get(model_id=model_id)

            # Check if model is deployed
            if model.status != dl.ModelStatus.DEPLOYED:
                logger.info(f"Model not deployed, deploying now | model_id={model_id}")
                model.deploy()

            # Execute model
            logger.info(f"Executing OCR model | model_id={model_id} item_id={item.id}")
            execution = model.predict(item_ids=[item.id])
            execution.wait()

            if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
                raise Exception(
                    f"OCR model execution failed: {execution.latest_status.get('message', 'Unknown error')}"
                )

            # Refresh item to get OCR results
            updated_item = dl.items.get(item_id=item.id)
            return OCRProcessor._parse_ocr_result(updated_item)

        except Exception as e:
            logger.error(f"Dataloop OCR model failed: {str(e)}")
            # Fallback to EasyOCR
            logger.info("Falling back to EasyOCR")
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)
                return OCRProcessor._extract_with_easyocr(file_path)

    @staticmethod
    def _extract_with_easyocr(image_path: str) -> str:
        """
        Extract text using EasyOCR.

        Args:
            image_path: Path to image file

        Returns:
            str: Extracted text
        """
        try:
            # Lazy import easyocr to avoid loading torch if not needed
            easyocr = _get_easyocr()

            # Initialize EasyOCR reader only once (class-level caching)
            if OCRProcessor._easyocr_reader is None:
                logger.info(f"Initializing EasyOCR reader with languages: {OCRProcessor._easyocr_languages}")
                OCRProcessor._easyocr_reader = easyocr.Reader(OCRProcessor._easyocr_languages, gpu=False)
                logger.info("EasyOCR reader initialized and cached")
            else:
                logger.debug("Using cached EasyOCR reader")

            # Perform OCR directly on image path
            resolved_path = str(Path(image_path).resolve())
            logger.debug(f"Running EasyOCR on image: {resolved_path}")
            results = OCRProcessor._easyocr_reader.readtext(resolved_path)

            # Extract text from results (bbox, text, confidence)
            all_text = ' '.join([text for (bbox, text, confidence) in results])

            logger.info(f"EasyOCR extracted {len(results)} text blocks, total length: {len(all_text)}")
            return all_text

        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return f"[OCR_ERROR: {str(e)}]"

    @staticmethod
    def _parse_ocr_result(item: dl.Item) -> str:
        """
        Parse OCR result from updated Dataloop item.
        Priority: item.description first, then text annotations.

        Args:
            item: Updated item after model execution

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
