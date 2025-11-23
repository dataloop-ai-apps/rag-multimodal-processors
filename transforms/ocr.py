"""
OCR enhancement transforms.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import logging
import os
import re
import tempfile
from typing import Dict, List

import dtlpy as dl

from utils.extracted_data import ExtractedData
from utils.dataloop_helpers import cleanup_temp_items_and_folder
from utils.ocr_utils import OCRProcessor

logger = logging.getLogger("rag-preprocessor")


def ocr_enhance(data: ExtractedData) -> ExtractedData:
    """
    Add OCR text from images to content.

    Args:
        data: ExtractedData with images and content_text

    Returns:
        ExtractedData with OCR text added to content
    """
    data.current_stage = "ocr"

    if not data.config.use_ocr:
        return data

    if not data.images:
        return data

    # Extract OCR text from all images, grouped by page
    ocr_by_page: Dict[int, List[str]] = {}

    for img in data.images:
        if img.path:
            try:
                ocr_text = OCRProcessor.extract_text_from_path(img.path)
                if ocr_text:
                    page_num = img.page_number or 0
                    if page_num not in ocr_by_page:
                        ocr_by_page[page_num] = []
                    ocr_by_page[page_num].append(ocr_text)
            except Exception as e:
                data.log_warning(f"OCR failed for {img.path}: {e}")

    if not ocr_by_page:
        return data

    # Integrate OCR text per page
    data.content_text = _integrate_ocr_per_page(data.content_text, ocr_by_page)

    # Add metadata
    total_ocr_length = sum(len(t) for texts in ocr_by_page.values() for t in texts)
    total_ocr_count = sum(len(texts) for texts in ocr_by_page.values())

    data.metadata['ocr_applied'] = True
    data.metadata['ocr_text_length'] = total_ocr_length
    data.metadata['ocr_image_count'] = total_ocr_count

    return data


def _integrate_ocr_per_page(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
    """
    Integrate OCR text into content on a per-page basis.

    Args:
        content: Original document content with page markers
        ocr_by_page: Dictionary mapping page numbers to lists of OCR texts

    Returns:
        Content with OCR text interleaved after each page
    """
    page_pattern = r'(--- Page (\d+) ---)'
    parts = re.split(page_pattern, content)

    if len(parts) <= 1:
        # No page markers found, append OCR at end
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            page_info = f" (Page {page_num})" if page_num else ""
            for ocr_text in ocr_by_page[page_num]:
                all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
        return content + '\n\n--- OCR Extracted Text ---\n\n' + '\n\n'.join(all_ocr_texts)

    # Reconstruct content with OCR text after each page
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


def describe_images(data: ExtractedData) -> ExtractedData:
    """
    Generate image descriptions using Dataloop vision models.

    Args:
        data: ExtractedData with images

    Returns:
        ExtractedData with image captions added
    """
    data.current_stage = "image_description"

    if not data.images:
        return data

    model_id = data.config.vision_model_id if hasattr(data.config, 'vision_model_id') else None
    if not model_id:
        data.log_warning("vision_model_id not provided, skipping image descriptions")
        return data

    try:
        model = dl.models.get(model_id=model_id)

        descriptions = []
        for img in data.images:
            if not img.path:
                continue

            try:
                result = model.predict([img.path])
                description = result[0] if result else "No description available"
                img.caption = description
                descriptions.append(description)
            except Exception as e:
                data.log_warning(f"Failed to describe image {img.path}: {e}")

        if descriptions:
            desc_text = '\n\n--- Image Descriptions ---\n\n' + '\n\n'.join(descriptions)
            data.content_text += desc_text

        data.metadata['image_descriptions_generated'] = True
        data.metadata['image_description_count'] = len(descriptions)

    except Exception as e:
        data.log_warning(f"Failed to generate image descriptions: {e}")

    return data


def ocr_batch_enhance(data: ExtractedData) -> ExtractedData:
    """
    Add OCR text using Dataloop batch processing.

    Args:
        data: ExtractedData with images and item

    Returns:
        ExtractedData with OCR text added
    """
    data.current_stage = "ocr_batch"

    if not data.config.use_ocr:
        return data

    if not data.images:
        return data

    model_id = data.config.ocr_model_id if hasattr(data.config, 'ocr_model_id') else None
    if not model_id:
        logger.warning("ocr_model_id not provided, falling back to local OCR")
        return ocr_enhance(data)

    if not data.item:
        data.log_error("Source item not provided, cannot perform batch OCR")
        return data

    try:
        logger.info(f"Starting batch OCR | model_id={model_id} image_count={len(data.images)}")

        temp_folder_name = f"./dataloop/temp_images_ocr_{data.item.name}"
        temp_local_dir = tempfile.mkdtemp(prefix=f"ocr_images_{data.item.id}_")

        uploaded_items = []
        try:
            uploaded_items = _upload_images_to_dataloop(
                data.images, temp_local_dir, temp_folder_name, data.item.dataset
            )

            if not uploaded_items:
                data.log_warning("No images were uploaded, skipping batch OCR")
                return data

            ocr_results = _run_batch_ocr_with_model(uploaded_items, model_id)

            # Store OCR results in images
            for ocr_result in ocr_results:
                page_num = ocr_result.get('page_number', 1)
                for img in data.images:
                    if img.page_number == page_num:
                        img.ocr_text = ocr_result.get('text', '')
                        break

            return ocr_enhance(data)

        finally:
            if uploaded_items:
                uploaded_item_objects = [item for item, _ in uploaded_items]
                cleanup_temp_items_and_folder(
                    uploaded_item_objects,
                    temp_folder_name,
                    data.item.dataset,
                    temp_local_dir
                )
            else:
                import shutil
                try:
                    shutil.rmtree(temp_local_dir, ignore_errors=True)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Batch OCR failed: {e}, falling back to local OCR")
        return ocr_enhance(data)


def _upload_images_to_dataloop(images, local_dir: str, remote_folder: str, dataset: dl.Dataset) -> list:
    """Upload images to Dataloop for batch processing."""
    if not images:
        return []

    image_paths = []
    image_metadata_list = []

    for idx, img in enumerate(images):
        try:
            if img.path and os.path.exists(img.path):
                image_paths.append(img.path)
                image_metadata_list.append({
                    'page_number': img.page_number or 1,
                    'image_index': idx,
                })
            else:
                logger.warning(f"Image path not found: {img.path}")
        except Exception as e:
            logger.warning(f"Failed to process image {idx}: {e}")

    if not image_paths:
        return []

    logger.info(f"Uploading {len(image_paths)} images | folder={remote_folder}")
    try:
        uploaded = dataset.items.upload(
            local_path=image_paths,
            remote_path=remote_folder,
            overwrite=True
        )

        if isinstance(uploaded, dl.Item):
            uploaded_items_list = [uploaded]
        else:
            uploaded_items_list = list(uploaded)

        uploaded_items = []
        for idx, uploaded_item in enumerate(uploaded_items_list):
            if idx < len(image_metadata_list):
                uploaded_items.append((uploaded_item, image_metadata_list[idx]))

        return uploaded_items

    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        return []


def _run_batch_ocr_with_model(uploaded_items: list, model_id: str) -> list:
    """Run batch OCR on uploaded items using Dataloop model."""
    if not uploaded_items:
        return []

    try:
        model = dl.models.get(model_id=model_id)

        if model.status != dl.ModelStatus.DEPLOYED:
            model.deploy()

        item_ids = [item.id for item, _ in uploaded_items]

        logger.info(f"Running batch OCR | model_id={model_id} items={len(item_ids)}")
        execution = model.predict(item_ids=item_ids)
        execution.wait()

        if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
            raise Exception(f"Batch OCR failed: {execution.latest_status.get('message')}")

        ocr_results = []
        for uploaded_item, metadata in uploaded_items:
            try:
                updated_item = dl.items.get(item_id=uploaded_item.id)
                ocr_text = OCRProcessor._parse_ocr_result(updated_item)
                ocr_results.append({
                    'page_number': metadata['page_number'],
                    'image_index': metadata['image_index'],
                    'text': ocr_text,
                })
            except Exception as e:
                logger.warning(f"Failed to get OCR for item {uploaded_item.id}: {e}")
                ocr_results.append({
                    'page_number': metadata['page_number'],
                    'image_index': metadata['image_index'],
                    'text': "",
                })

        return ocr_results

    except Exception as e:
        logger.error(f"Batch OCR prediction failed: {e}")
        raise
