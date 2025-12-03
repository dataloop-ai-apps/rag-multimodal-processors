"""
OCR enhancement transforms.

All functions follow signature: (data: ExtractedData) -> ExtractedData

NOTE: Dataloop model integration for batch OCR and image captioning is not yet implemented.
Currently only local EasyOCR is supported.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.extracted_data import ExtractedData
from utils.data_types import ImageContent

warnings.filterwarnings(
    'ignore', category=DeprecationWarning, module='torch.ao.quantization', message='.*torch.ao.quantization.*'
)

_easyocr = None


def _get_easyocr():
    """Lazy import easyocr to avoid loading torch until needed."""
    global _easyocr
    if _easyocr is None:
        import easyocr
        _easyocr = easyocr
    return _easyocr


logger = logging.getLogger("rag-preprocessor")


class OCREnhancer:
    """OCR text extraction using local EasyOCR."""

    _easyocr_reader = None
    _easyocr_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']

    @staticmethod
    def extract_local(images: List[ImageContent]) -> Tuple[Dict[int, List[str]], Dict[str, str]]:
        """
        Extract OCR text from images using local EasyOCR.

        Returns:
            Tuple of (ocr_by_page dict, errors dict mapping image_path -> error_message)
        """
        ocr_by_page: Dict[int, List[str]] = {}
        ocr_errors: Dict[str, str] = {}

        for img in images:
            if img.path:
                try:
                    ocr_text, error_msg = OCREnhancer._extract_with_easyocr(img.path)
                    if error_msg:
                        ocr_errors[img.path] = error_msg
                    elif ocr_text:
                        page_num = img.page_number or 0
                        if page_num not in ocr_by_page:
                            ocr_by_page[page_num] = []
                        ocr_by_page[page_num].append(ocr_text)
                except Exception as e:
                    error_message = f"OCR failed for {img.path}: {e}"
                    logger.warning(error_message)
                    ocr_errors[img.path] = str(e)

        return ocr_by_page, ocr_errors

    @staticmethod
    def extract_batch(
        images: List[ImageContent], model_id: str, dataset, item_name: str, item_id: str
    ) -> Tuple[Dict[int, List[str]], Dict[str, str]]:
        """
        Extract OCR text from images using Dataloop batch processing.

        TODO: Implement Dataloop model integration.
        Currently falls back to local OCR.

        Returns:
            Tuple of (ocr_by_page dict, errors dict mapping image_path -> error_message)
        """
        logger.warning("Batch OCR with Dataloop models not yet implemented. Using local OCR.")
        return OCREnhancer.extract_local(images)

    @staticmethod
    def integrate_ocr_append_to_page(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
        """Integrate OCR text by appending to each page."""
        page_pattern = r'(--- Page (\d+) ---)'
        parts = re.split(page_pattern, content)

        if len(parts) <= 1:
            all_ocr_texts = []
            for page_num in sorted(ocr_by_page.keys()):
                page_info = f" (Page {page_num})" if page_num else ""
                for ocr_text in ocr_by_page[page_num]:
                    all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
            return content + '\n\n--- OCR Extracted Text ---\n\n' + '\n\n'.join(all_ocr_texts)

        result_parts = []

        if parts[0].strip():
            result_parts.append(parts[0])

        i = 1
        while i < len(parts):
            if i + 2 < len(parts):
                page_marker = parts[i]
                page_num = int(parts[i + 1])
                page_content = parts[i + 2]

                result_parts.append(page_marker)
                result_parts.append(page_content)

                if page_num in ocr_by_page and ocr_by_page[page_num]:
                    ocr_section = [f"\n--- OCR from Page {page_num} Images ---"]
                    for idx, ocr_text in enumerate(ocr_by_page[page_num], 1):
                        ocr_section.append(f"\nImage {idx}:\n{ocr_text}")
                    result_parts.append('\n'.join(ocr_section))

                i += 3
            else:
                result_parts.extend(parts[i:])
                break

        return ''.join(result_parts)

    @staticmethod
    def integrate_ocr_separate_chunks(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
        """Integrate OCR text as a separate section at the end."""
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            if ocr_by_page[page_num]:
                for idx, ocr_text in enumerate(ocr_by_page[page_num], 1):
                    page_info = f" (Page {page_num})" if page_num else ""
                    all_ocr_texts.append(f"--- Image {idx}{page_info} ---\n{ocr_text}")

        if all_ocr_texts:
            return content + '\n\n--- OCR Extracted Text (Separate Section) ---\n\n' + '\n\n'.join(all_ocr_texts)
        return content

    @staticmethod
    def integrate_ocr_combine_all(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
        """Integrate OCR text by combining all at the end without page markers."""
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            all_ocr_texts.extend(ocr_by_page[page_num])

        if all_ocr_texts:
            combined_ocr = '\n\n'.join(all_ocr_texts)
            return content + '\n\n--- All OCR Text Combined ---\n\n' + combined_ocr
        return content

    @staticmethod
    def integrate_ocr_per_page(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
        """Legacy method - calls integrate_ocr_append_to_page for backward compatibility."""
        return OCREnhancer.integrate_ocr_append_to_page(content, ocr_by_page)

    @staticmethod
    def _extract_with_easyocr(image_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text using EasyOCR.

        Returns:
            Tuple of (extracted_text, error_message)
            - If successful: (text, None)
            - If failed: ("", error_message_string)
        """
        try:
            easyocr = _get_easyocr()
            if OCREnhancer._easyocr_reader is None:
                logger.info(f"Initializing EasyOCR reader with languages: {OCREnhancer._easyocr_languages}")
                OCREnhancer._easyocr_reader = easyocr.Reader(OCREnhancer._easyocr_languages, gpu=False)
                logger.info("EasyOCR reader initialized and cached")

            resolved_path = str(Path(image_path).resolve())
            results = OCREnhancer._easyocr_reader.readtext(resolved_path)
            all_text = ' '.join([text for (bbox, text, confidence) in results])

            logger.info(f"EasyOCR extracted {len(results)} text blocks, total length: {len(all_text)}")
            return all_text, None

        except Exception as e:
            error_message = f"EasyOCR failed: {str(e)}"
            logger.error(error_message)
            return "", error_message


class ImageDescriber:
    """
    Image description using vision models.
    
    TODO: Implement Dataloop model integration.
    """

    @staticmethod
    def describe(images: List[ImageContent], model_id: str, dataset, item_name: str, item_id: str) -> List[str]:
        """
        Generate descriptions for images using a vision model.

        TODO: Implement Dataloop model integration.
        """
        logger.warning("Image captioning not yet implemented")
        return []


# Transform wrappers


def ocr_enhance(data: ExtractedData) -> ExtractedData:
    """
    Add OCR text from images to content.

    Currently only 'local' method (EasyOCR) is implemented.
    Batch and auto methods fall back to local.
    """
    data.current_stage = "ocr"

    if not data.config.ocr_from_images:
        return data

    if not data.images:
        return data

    if data.config.ocr_method in ('batch', 'auto'):
        logger.info(f"OCR method '{data.config.ocr_method}' requested but batch not implemented. Using local OCR.")

    ocr_by_page, ocr_errors = OCREnhancer.extract_local(data.images)

    # Store OCR errors in metadata
    if ocr_errors:
        data.metadata['ocr_errors'] = ocr_errors
        data.metadata['ocr_failed_count'] = len(ocr_errors)
        # Log warnings for each failed OCR attempt
        for image_path, error_msg in ocr_errors.items():
            data.log_warning(f"EasyOCR failed for {image_path}: {error_msg}")

    if not ocr_by_page:
        return data

    # Use the configured OCR integration method
    integration_method = data.config.ocr_integration_method
    if integration_method == 'separate_chunks':
        data.content_text = OCREnhancer.integrate_ocr_separate_chunks(data.content_text, ocr_by_page)
    elif integration_method == 'combine_all':
        data.content_text = OCREnhancer.integrate_ocr_combine_all(data.content_text, ocr_by_page)
    else:  # Default to 'append_to_page'
        data.content_text = OCREnhancer.integrate_ocr_append_to_page(data.content_text, ocr_by_page)

    total_ocr_length = sum(len(t) for texts in ocr_by_page.values() for t in texts)
    total_ocr_count = sum(len(texts) for texts in ocr_by_page.values())

    data.metadata['ocr_applied'] = True
    data.metadata['ocr_method'] = 'local'
    data.metadata['ocr_integration_method'] = integration_method
    data.metadata['ocr_text_length'] = total_ocr_length
    data.metadata['ocr_image_count'] = total_ocr_count

    return data


def describe_images(data: ExtractedData) -> ExtractedData:
    """
    Generate image descriptions using vision models.

    TODO: Implement Dataloop model integration.
    Currently returns without modifications.
    """
    data.current_stage = "image_description"

    if not data.images:
        return data

    if not data.config.vision_model_id:
        data.log_warning("Vision model not configured. Skipping image descriptions.")
        return data

    # Placeholder - not yet implemented
    data.log_warning("Image captioning with Dataloop models not yet implemented. Skipping.")
    data.metadata['image_descriptions_generated'] = False

    return data
