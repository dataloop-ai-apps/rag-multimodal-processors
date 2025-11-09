"""
OCR enhancement stages.
Adds OCR-extracted text to content.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any, List


def ocr_enhance(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add OCR text from images to content.

    Args:
        data: Must contain 'images' list and 'content' string
        config: Can contain 'use_ocr', 'ocr_integration_method'

    Returns:
        data with OCR text added to content
    """
    if not config.get('use_ocr', False):
        return data

    images = data.get('images', [])
    if not images:
        return data

    try:
        from extractors.ocr_extractor import OCRExtractor
    except ImportError:
        print("Warning: OCRExtractor not found, skipping OCR")
        return data

    ocr_extractor = OCRExtractor()
    ocr_texts = []

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
                    page_info = f" (Page {page_num})" if page_num else ""
                    ocr_texts.append(f"--- Image{page_info} ---\n{ocr_text}")
            except Exception as e:
                print(f"Warning: OCR failed for {img_path}: {e}")

    if ocr_texts:
        integration_method = config.get('ocr_integration_method', 'append')

        if integration_method == 'append':
            data['content'] += '\n\n--- OCR Extracted Text ---\n\n' + '\n\n'.join(ocr_texts)
        elif integration_method == 'prepend':
            data['content'] = '\n\n'.join(ocr_texts) + '\n\n--- Original Text ---\n\n' + data['content']
        elif integration_method == 'separate':
            data['ocr_content'] = '\n\n'.join(ocr_texts)

        data.setdefault('metadata', {})['ocr_applied'] = True
        data['metadata']['ocr_text_length'] = sum(len(t) for t in ocr_texts)
        data['metadata']['ocr_image_count'] = len(ocr_texts)

    return data


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

    import dtlpy as dl

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

    import dtlpy as dl

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
