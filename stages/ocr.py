"""
OCR enhancement stages.
Adds OCR-extracted text to content.
All functions follow signature: (data: dict, config: dict) -> dict
"""

import re
import dtlpy as dl
from typing import Dict, Any, List
from extractors.ocr_extractor import OCRExtractor


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

    ocr_extractor = OCRExtractor()

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
                ocr_text = ocr_extractor.extract_text(img_path)
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
    Useful for processing many images at once.

    Args:
        data: Must contain 'images' list
        config: Must contain 'ocr_model_id' for Dataloop OCR model

    Returns:
        data with OCR text added
    """
    if not config.get('use_dataloop_ocr', False):
        return data

    images = data.get('images', [])
    if not images:
        return data

    model_id = config.get('ocr_model_id')
    if not model_id:
        print("Warning: ocr_model_id not provided, skipping batch OCR")
        return data

    try:
        # Upload images as items and run batch OCR
        # This would require implementing batch processing logic
        # For now, fall back to single-image OCR
        print("Warning: Batch OCR not implemented, falling back to single-image OCR")
        return ocr_enhance(data, config)

    except Exception as e:
        print(f"Warning: Batch OCR failed: {e}")

    return data
