"""
OCR enhancement stages.
Adds OCR-extracted text to content.
All functions follow signature: (data: dict, config: dict) -> dict
"""

import logging
import os
import re
import tempfile
from typing import Dict, Any, List

import dtlpy as dl

from utils.dataloop_helpers import cleanup_temp_items_and_folder
from utils.ocr_utils import OCRProcessor

logger = logging.getLogger("rag-preprocessor")


def ocr_enhance(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add OCR text from images to content.

    Args:
        data: Must contain 'images' list and 'content' string
        config: Can contain:
            - 'use_ocr' (bool): Enable OCR processing
            - 'ocr_integration_method' (str): How to integrate OCR text
              Options: 'per_page', 'append', 'prepend', 'separate'

    Returns:
        data with OCR text added to content

    Integration methods:
        - 'per_page': Interleave OCR text after each page's content (maintains structure)
        - 'append': Add all OCR text at the end
        - 'prepend': Add all OCR text at the beginning
        - 'separate': Store OCR text in separate field 'ocr_content'
    """
    if not config.get('use_ocr', False):
        return data

    images = data.get('images', [])
    if not images:
        return data

    # Extract OCR text from all images, grouped by page
    ocr_by_page = {}  # {page_number: [ocr_texts]}

    for img in images:
        if isinstance(img, dict):
            img_path = img.get('path')
            page_num = img.get('page_number')
        else:
            img_path = img.path if hasattr(img, 'path') else str(img)
            page_num = img.page_number if hasattr(img, 'page_number') else None

        if img_path:
            try:
                # Use OCRProcessor for local file paths
                ocr_text = OCRProcessor.extract_text_from_path(img_path)
                if ocr_text:
                    if page_num not in ocr_by_page:
                        ocr_by_page[page_num] = []
                    ocr_by_page[page_num].append(ocr_text)
            except Exception as e:
                print(f"Warning: OCR failed for {img_path}: {e}")

    if not ocr_by_page:
        return data

    integration_method = config.get('ocr_integration_method', 'per_page')

    if integration_method == 'per_page':
        # Interleave OCR text after each page's content
        data['content'] = _integrate_ocr_per_page(data['content'], ocr_by_page)

    elif integration_method == 'append':
        # Add all OCR text at the end
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            page_info = f" (Page {page_num})" if page_num else ""
            for ocr_text in ocr_by_page[page_num]:
                all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
        data['content'] += '\n\n--- OCR Extracted Text ---\n\n' + '\n\n'.join(all_ocr_texts)

    elif integration_method == 'prepend':
        # Add all OCR text at the beginning
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            page_info = f" (Page {page_num})" if page_num else ""
            for ocr_text in ocr_by_page[page_num]:
                all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
        data['content'] = '\n\n'.join(all_ocr_texts) + '\n\n--- Original Text ---\n\n' + data['content']

    elif integration_method == 'separate':
        # Store OCR text in separate field
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            page_info = f" (Page {page_num})" if page_num else ""
            for ocr_text in ocr_by_page[page_num]:
                all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
        data['ocr_content'] = '\n\n'.join(all_ocr_texts)

    # Add metadata
    total_ocr_length = sum(len(t) for texts in ocr_by_page.values() for t in texts)
    total_ocr_count = sum(len(texts) for texts in ocr_by_page.values())

    data.setdefault('metadata', {})['ocr_applied'] = True
    data['metadata']['ocr_text_length'] = total_ocr_length
    data['metadata']['ocr_image_count'] = total_ocr_count
    data['metadata']['ocr_integration_method'] = integration_method

    return data


def _integrate_ocr_per_page(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
    """
    Integrate OCR text into content on a per-page basis.

    Parses content by page markers and inserts OCR text after each page.

    Args:
        content: Original document content with page markers
        ocr_by_page: Dictionary mapping page numbers to lists of OCR texts

    Returns:
        Content with OCR text interleaved after each page
    """
    # Split content by page markers: "--- Page N ---"
    page_pattern = r'(--- Page (\d+) ---)'
    parts = re.split(page_pattern, content)

    if len(parts) <= 1:
        # No page markers found, fall back to append method
        all_ocr_texts = []
        for page_num in sorted(ocr_by_page.keys()):
            page_info = f" (Page {page_num})" if page_num else ""
            for ocr_text in ocr_by_page[page_num]:
                all_ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
        return content + '\n\n--- OCR Extracted Text ---\n\n' + '\n\n'.join(all_ocr_texts)

    # Reconstruct content with OCR text after each page
    result_parts = []

    # First part before any page markers
    if parts[0].strip():
        result_parts.append(parts[0])

    # Process each page
    i = 1
    while i < len(parts):
        if i + 2 < len(parts):
            # parts[i] = full marker "--- Page N ---"
            # parts[i+1] = page number "N"
            # parts[i+2] = page content

            page_marker = parts[i]
            page_num = int(parts[i + 1])
            page_content = parts[i + 2]

            # Add page marker and content
            result_parts.append(page_marker)
            result_parts.append(page_content)

            # Add OCR text for this page if available
            if page_num in ocr_by_page and ocr_by_page[page_num]:
                ocr_section = [f"\n--- OCR from Page {page_num} Images ---"]
                for idx, ocr_text in enumerate(ocr_by_page[page_num], 1):
                    ocr_section.append(f"\nImage {idx}:\n{ocr_text}")
                result_parts.append('\n'.join(ocr_section))

            i += 3
        else:
            # Remaining parts
            result_parts.extend(parts[i:])
            break

    return ''.join(result_parts)


def describe_images_with_dataloop(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate image descriptions using Dataloop vision models.

    Args:
        data: Must contain 'images' list
        config: Must contain 'vision_model_id' for Dataloop model

    Returns:
        data with image captions added
    """
    if not config.get('describe_images', False):
        return data

    images = data.get('images', [])
    if not images:
        return data

    model_id = config.get('vision_model_id')
    if not model_id:
        print("Warning: vision_model_id not provided, skipping image descriptions")
        return data

    try:
        model = dl.models.get(model_id=model_id)

        descriptions = []
        for img in images:
            img_path = img.get('path') if isinstance(img, dict) else (img.path if hasattr(img, 'path') else None)

            if not img_path:
                continue

            try:
                # NOTE: According to Dataloop SDK, model.predict() only accepts item_ids or dataset_id
                # and returns an Execution object, not direct results. This code may need to be updated
                # to create items first and use item_ids, then wait for execution and retrieve results.
                # Run vision model
                result = model.predict([img_path])
                description = result[0] if result else "No description available"

                # Update image object
                if isinstance(img, dict):
                    img['caption'] = description

                descriptions.append(description)
            except Exception as e:
                print(f"Warning: Failed to describe image {img_path}: {e}")

        # Optionally add descriptions to content
        if config.get('include_image_descriptions_in_content', True) and descriptions:
            desc_text = '\n\n--- Image Descriptions ---\n\n' + '\n\n'.join(descriptions)
            data['content'] += desc_text

        data.setdefault('metadata', {})['image_descriptions_generated'] = True
        data['metadata']['image_description_count'] = len(descriptions)

    except Exception as e:
        print(f"Warning: Failed to generate image descriptions: {e}")

    return data


def ocr_batch_enhance(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add OCR text using Dataloop batch processing.
    Uploads images to Dataloop, runs batch OCR model, retrieves results, and cleans up.

    Args:
        data: Must contain 'images' list and 'item' (source item)
        config: Must contain 'custom_ocr_model_id' for Dataloop OCR model

    Returns:
        data with OCR text added via ocr_enhance integration
    """
    if not config.get('use_dataloop_ocr', False):
        return data

    images = data.get('images', [])
    if not images:
        return data

    model_id = config.get('custom_ocr_model_id')
    if not model_id:
        logger.warning("custom_ocr_model_id not provided, falling back to EasyOCR")
        return ocr_enhance(data, config)

    item = data.get('item')
    if not item:
        logger.error("Source item not provided in data, cannot perform batch OCR")
        return data

    try:
        logger.info(f"Starting batch OCR with Dataloop model | model_id={model_id} image_count={len(images)}")

        # Setup temp directories
        temp_folder_name = f"./dataloop/temp_images_ocr_{item.name}"
        temp_local_dir = tempfile.mkdtemp(prefix=f"ocr_images_{item.id}_")

        uploaded_items = []
        try:
            # Upload images to Dataloop
            uploaded_items = _upload_images_to_dataloop(
                images, temp_local_dir, temp_folder_name, item.dataset
            )

            if not uploaded_items:
                logger.warning("No images were uploaded, skipping batch OCR")
                return data

            # Run batch OCR
            ocr_results = _run_batch_ocr_with_model(uploaded_items, model_id)

            # Convert results to format expected by ocr_enhance
            # Update images with OCR text
            for ocr_result in ocr_results:
                page_num = ocr_result.get('page_number', 1)
                # Find matching image and add OCR text as a temporary file
                for img in images:
                    img_page = img.get('page_number') if isinstance(img, dict) else getattr(img, 'page_number', None)
                    if img_page == page_num:
                        # Store OCR text in image metadata so ocr_enhance can use it
                        if isinstance(img, dict):
                            img['ocr_text'] = ocr_result.get('text', '')
                        else:
                            setattr(img, 'ocr_text', ocr_result.get('text', ''))
                        break

            # Now use ocr_enhance to integrate the OCR text
            return ocr_enhance(data, config)

        finally:
            # Cleanup all temporary resources
            if uploaded_items:
                uploaded_item_objects = [item for item, _ in uploaded_items]
                cleanup_temp_items_and_folder(
                    uploaded_item_objects,
                    temp_folder_name,
                    item.dataset,
                    temp_local_dir
                )
            else:
                # Clean up local dir if no items were uploaded
                import shutil
                try:
                    shutil.rmtree(temp_local_dir, ignore_errors=True)
                except:
                    pass

    except Exception as e:
        logger.error(f"Batch OCR failed: {e}, falling back to single-image OCR")
        return ocr_enhance(data, config)


def _upload_images_to_dataloop(
    images: List[Any],
    local_dir: str,
    remote_folder: str,
    dataset: dl.Dataset
) -> List[tuple]:
    """
    Save images locally and upload to Dataloop in batch.

    Args:
        images: List of ImageContent objects or dicts with 'path', 'page_number'
        local_dir: Local temporary directory
        remote_folder: Remote folder path in dataset
        dataset: Target dataset

    Returns:
        List of tuples: (uploaded_item, metadata)
    """
    if not images:
        return []

    # Step 1: Collect image paths and metadata
    image_paths = []
    image_metadata_list = []

    for idx, img in enumerate(images):
        try:
            # Get image path and page number
            if isinstance(img, dict):
                img_path = img.get('path')
                page_num = img.get('page_number', 1)
            else:
                img_path = img.path if hasattr(img, 'path') else None
                page_num = img.page_number if hasattr(img, 'page_number') else 1

            if img_path and os.path.exists(img_path):
                image_paths.append(img_path)
                image_metadata_list.append({
                    'page_number': page_num,
                    'image_index': idx,
                })
            else:
                logger.warning(f"Image path not found or invalid: {img_path}")

        except Exception as e:
            logger.warning(f"Failed to process image {idx}: {e}")
            continue

    if not image_paths:
        logger.warning("No valid images to upload")
        return []

    # Step 2: Batch upload to Dataloop
    logger.info(f"Batch uploading {len(image_paths)} images to Dataloop | folder={remote_folder}")
    try:
        uploaded = dataset.items.upload(
            local_path=image_paths,
            remote_path=remote_folder,
            overwrite=True
        )

        # Handle single item vs list response
        if isinstance(uploaded, dl.Item):
            uploaded_items_list = [uploaded]
        else:
            uploaded_items_list = list(uploaded)

        # Pair uploaded items with their metadata
        uploaded_items = []
        for idx, uploaded_item in enumerate(uploaded_items_list):
            if idx < len(image_metadata_list):
                uploaded_items.append((uploaded_item, image_metadata_list[idx]))
                logger.debug(f"Uploaded image | item_id={uploaded_item.id} page={image_metadata_list[idx]['page_number']}")

        logger.info(f"Batch upload completed | uploaded={len(uploaded_items)} images")
        return uploaded_items

    except Exception as e:
        logger.error(f"Batch upload failed: {e}, falling back to individual uploads")
        uploaded_items = []
        for image_path, metadata in zip(image_paths, image_metadata_list):
            try:
                uploaded_item = dataset.items.upload(
                    local_path=image_path,
                    remote_path=remote_folder,
                    overwrite=True
                )
                uploaded_items.append((uploaded_item, metadata))
            except Exception as upload_err:
                logger.warning(f"Failed to upload {image_path}: {upload_err}")
                continue

        logger.info(f"Individual uploads completed | uploaded={len(uploaded_items)} images")
        return uploaded_items


def _run_batch_ocr_with_model(
    uploaded_items: List[tuple],
    model_id: str
) -> List[Dict[str, Any]]:
    """
    Run batch OCR on uploaded items using Dataloop model.

    Args:
        uploaded_items: List of (item, metadata) tuples
        model_id: Dataloop OCR model ID

    Returns:
        List of OCR results with page numbers and text
    """
    if not uploaded_items:
        return []

    try:
        # Get the model
        model = dl.models.get(model_id=model_id)

        # Check if model is deployed
        if model.status != dl.ModelStatus.DEPLOYED:
            logger.info(f"Model not deployed, deploying now | model_id={model_id}")
            model.deploy()

        # Extract item objects
        uploaded_item_objects = [item for item, _ in uploaded_items]
        item_ids = [item.id for item in uploaded_item_objects]

        # Execute batch prediction
        logger.info(f"Running batch OCR prediction | model_id={model_id} item_count={len(item_ids)}")
        execution = model.predict(item_ids=item_ids)
        logger.info(f"Waiting for batch execution to complete | execution_id={execution.id}")
        execution.wait()

        # Check execution status
        if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
            raise Exception(f"Batch OCR execution failed: {execution.latest_status.get('message', 'Unknown error')}")
        elif execution.latest_status['status'] in ['success', dl.ExecutionStatus.SUCCESS]:
            logger.info(f"Batch OCR execution successful | execution_id={execution.id}")

        # Fetch results from each item
        ocr_results = []
        for uploaded_item, metadata in uploaded_items:
            try:
                # Refresh item to get OCR results
                updated_item = dl.items.get(item_id=uploaded_item.id)
                ocr_text = OCRProcessor._parse_ocr_result(updated_item)

                ocr_results.append({
                    'page_number': metadata['page_number'],
                    'image_index': metadata['image_index'],
                    'text': ocr_text,
                })
                logger.debug(f"OCR text retrieved | item_id={uploaded_item.id} text_length={len(ocr_text)}")
            except Exception as e:
                logger.warning(f"Failed to retrieve OCR results for item {uploaded_item.id}: {e}")
                ocr_results.append({
                    'page_number': metadata['page_number'],
                    'image_index': metadata['image_index'],
                    'text': "",
                })

        success_count = sum(1 for r in ocr_results if r['text'])
        logger.info(f"Batch OCR completed | successful={success_count}/{len(uploaded_items)}")
        return ocr_results

    except Exception as e:
        logger.error(f"Batch OCR prediction failed: {str(e)}")
        raise
