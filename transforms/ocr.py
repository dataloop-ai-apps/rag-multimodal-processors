"""
OCR enhancement transforms.

Public API:
- ocr_enhance(): Single entry point for OCR text extraction
- describe_images(): Generate image captions using vision models

All functions follow signature: (data: ExtractedData) -> ExtractedData

Note: EasyOCR uses torch.ao.quantization which is deprecated in PyTorch 2.10+.
See: https://github.com/pytorch/ao/issues/2259
"""

import logging
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import dtlpy as dl

from utils.extracted_data import ExtractedData
from utils.dataloop_helpers import cleanup_temp_items_and_folder
from utils.data_types import ImageContent

warnings.filterwarnings(
    'ignore', category=DeprecationWarning, module='torch.ao.quantization', message='.*torch.ao.quantization.*'
)

_easyocr = None


def _get_easyocr():
    """Lazy import easyocr to avoid loading heavy dependencies until needed."""
    global _easyocr
    if _easyocr is None:
        import easyocr as _easyocr_module

        _easyocr = _easyocr_module
    return _easyocr


logger = logging.getLogger("rag-preprocessor")


class OCREnhancer:
    """OCR text extraction supporting local EasyOCR and Dataloop models."""

    _easyocr_reader = None
    _easyocr_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']

    @staticmethod
    def extract_local(images: List[ImageContent]) -> Dict[int, List[str]]:
        """Extract OCR text from images using local EasyOCR."""
        ocr_by_page: Dict[int, List[str]] = {}

        for img in images:
            if img.path:
                try:
                    ocr_text = OCREnhancer._extract_with_easyocr(img.path)
                    if ocr_text:
                        page_num = img.page_number or 0
                        if page_num not in ocr_by_page:
                            ocr_by_page[page_num] = []
                        ocr_by_page[page_num].append(ocr_text)
                except Exception as e:
                    logger.warning(f"OCR failed for {img.path}: {e}")

        return ocr_by_page

    @staticmethod
    def extract_text_from_path(image_path: str) -> str:
        """Extract text from local image file path using EasyOCR."""
        return OCREnhancer._extract_with_easyocr(image_path)

    @staticmethod
    def extract_with_model(item: dl.Item, config: Dict[str, Any]) -> str:
        """Extract OCR text using Dataloop model with EasyOCR fallback."""
        model_id = config.get('custom_ocr_model_id')

        if model_id:
            return OCREnhancer._extract_with_dataloop_model(item, model_id)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)
                return OCREnhancer._extract_with_easyocr(file_path)

    @staticmethod
    def extract_batch(
        images: List[ImageContent], model_id: str, dataset: dl.Dataset, item_name: str, item_id: str
    ) -> Dict[int, List[str]]:
        """Extract OCR text from images using Dataloop batch processing."""
        logger.info(f"Starting batch OCR | model_id={model_id} image_count={len(images)}")

        temp_folder_name = f"./dataloop/temp_images_ocr_{item_name}"
        temp_local_dir = tempfile.mkdtemp(prefix=f"ocr_images_{item_id}_")

        uploaded_items = []
        try:
            uploaded_items = OCREnhancer._upload_images_to_dataloop(images, temp_local_dir, temp_folder_name, dataset)

            if not uploaded_items:
                logger.warning("No images were uploaded for batch OCR")
                return {}

            ocr_results = OCREnhancer._run_batch_ocr_with_model(uploaded_items, model_id)

            ocr_by_page: Dict[int, List[str]] = {}
            for ocr_result in ocr_results:
                page_num = ocr_result.get('page_number', 0)
                text = ocr_result.get('text', '')
                if text:
                    if page_num not in ocr_by_page:
                        ocr_by_page[page_num] = []
                    ocr_by_page[page_num].append(text)

            return ocr_by_page

        finally:
            OCREnhancer._cleanup_batch_resources(uploaded_items, temp_folder_name, dataset, temp_local_dir)

    @staticmethod
    def integrate_ocr_per_page(content: str, ocr_by_page: Dict[int, List[str]]) -> str:
        """Integrate OCR text into content on a per-page basis."""
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

    # Private EasyOCR methods

    @staticmethod
    def _extract_with_easyocr(image_path: str) -> str:
        """Extract text using EasyOCR."""
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
            return all_text

        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return f"[OCR_ERROR: {str(e)}]"

    # Private Dataloop model methods

    @staticmethod
    def _extract_with_dataloop_model(item: dl.Item, model_id: str) -> str:
        """Extract text using Dataloop OCR model."""
        try:
            model = dl.models.get(model_id=model_id)

            if model.status != dl.ModelStatus.DEPLOYED:
                logger.info(f"Model not deployed, deploying now | model_id={model_id}")
                model.deploy()

            logger.info(f"Executing OCR model | model_id={model_id} item_id={item.id}")
            execution = model.predict(item_ids=[item.id])
            execution.wait()

            if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
                raise Exception(
                    f"OCR model execution failed: {execution.latest_status.get('message', 'Unknown error')}"
                )

            updated_item = dl.items.get(item_id=item.id)
            return OCREnhancer._parse_ocr_result(updated_item)

        except Exception as e:
            logger.error(f"Dataloop OCR model failed: {str(e)}")
            logger.info("Falling back to EasyOCR")
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)
                return OCREnhancer._extract_with_easyocr(file_path)

    @staticmethod
    def _parse_ocr_result(item: dl.Item) -> str:
        """Parse OCR result from updated Dataloop item."""
        extracted_text = ""

        if item.description and item.description.strip():
            logger.info(f"Found OCR text in item.description | length={len(item.description)}")
            extracted_text = item.description.strip()
        else:
            try:
                annotations = item.annotations.list()
                text_annotations = []

                for annotation in annotations:
                    if hasattr(annotation, 'label') and annotation.label == 'Text':
                        if hasattr(annotation, 'description') and annotation.description:
                            text_annotations.append(annotation.description)

                if text_annotations:
                    extracted_text = ' '.join(text_annotations)
                    logger.info(f"Found {len(text_annotations)} text annotations | total_length={len(extracted_text)}")
                else:
                    logger.warning(f"No OCR results found for item {item.id}")
            except Exception as e:
                logger.warning(f"Failed to parse annotations: {str(e)}")

        return extracted_text

    # Private batch processing methods

    @staticmethod
    def _upload_images_to_dataloop(
        images: List[ImageContent], local_dir: str, remote_folder: str, dataset: dl.Dataset
    ) -> List[Tuple[dl.Item, dict]]:
        """Upload images to Dataloop for batch processing."""
        if not images:
            return []

        image_paths = []
        image_metadata_list = []

        for idx, img in enumerate(images):
            try:
                if img.path and os.path.exists(img.path):
                    image_paths.append(img.path)
                    image_metadata_list.append({'page_number': img.page_number or 1, 'image_index': idx})
                else:
                    logger.warning(f"Image path not found: {img.path}")
            except Exception as e:
                logger.warning(f"Failed to process image {idx}: {e}")

        if not image_paths:
            return []

        logger.info(f"Uploading {len(image_paths)} images | folder={remote_folder}")
        try:
            uploaded = dataset.items.upload(local_path=image_paths, remote_path=remote_folder, overwrite=True)

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

    @staticmethod
    def _run_batch_ocr_with_model(uploaded_items: List[Tuple[dl.Item, dict]], model_id: str) -> List[dict]:
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
                    ocr_text = OCREnhancer._parse_ocr_result(updated_item)
                    ocr_results.append(
                        {
                            'page_number': metadata['page_number'],
                            'image_index': metadata['image_index'],
                            'text': ocr_text,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to get OCR for item {uploaded_item.id}: {e}")
                    ocr_results.append(
                        {'page_number': metadata['page_number'], 'image_index': metadata['image_index'], 'text': ""}
                    )

            return ocr_results

        except Exception as e:
            logger.error(f"Batch OCR prediction failed: {e}")
            raise

    @staticmethod
    def _cleanup_batch_resources(
        uploaded_items: List[Tuple[dl.Item, dict]], temp_folder_name: str, dataset: dl.Dataset, temp_local_dir: str
    ) -> None:
        """Clean up temporary resources from batch OCR."""
        if uploaded_items:
            uploaded_item_objects = [item for item, _ in uploaded_items]
            cleanup_temp_items_and_folder(uploaded_item_objects, temp_folder_name, dataset, temp_local_dir)
        else:
            try:
                shutil.rmtree(temp_local_dir, ignore_errors=True)
            except Exception:
                pass


class ImageDescriber:
    """Image description operations using vision models."""

    @staticmethod
    def describe(
        images: List[ImageContent], model_id: str, dataset: dl.Dataset, item_name: str, item_id: str
    ) -> List[str]:
        """Generate descriptions for images using a Dataloop vision model."""
        logger.info(f"Starting image description | model_id={model_id} image_count={len(images)}")

        temp_folder_name = f"./dataloop/temp_images_caption_{item_name}"
        temp_local_dir = tempfile.mkdtemp(prefix=f"caption_images_{item_id}_")

        uploaded_items = []
        try:
            uploaded_items = ImageDescriber._upload_images_to_dataloop(
                images, temp_local_dir, temp_folder_name, dataset
            )

            if not uploaded_items:
                logger.warning("No images were uploaded for captioning")
                return []

            descriptions = ImageDescriber._run_captioning_model(uploaded_items, model_id, images, dataset)

            return descriptions

        finally:
            ImageDescriber._cleanup_resources(uploaded_items, temp_folder_name, dataset, temp_local_dir)

    @staticmethod
    def _upload_images_to_dataloop(
        images: List[ImageContent], local_dir: str, remote_folder: str, dataset: dl.Dataset
    ) -> List[Tuple[dl.Item, dict]]:
        """Upload images to Dataloop for captioning."""
        if not images:
            return []

        image_paths = []
        image_metadata_list = []

        for idx, img in enumerate(images):
            try:
                if img.path and os.path.exists(img.path):
                    image_paths.append(img.path)
                    image_metadata_list.append({'page_number': img.page_number or 1, 'image_index': idx})
                else:
                    logger.warning(f"Image path not found: {img.path}")
            except Exception as e:
                logger.warning(f"Failed to process image {idx}: {e}")

        if not image_paths:
            return []

        logger.info(f"Uploading {len(image_paths)} images for captioning | folder={remote_folder}")
        try:
            uploaded = dataset.items.upload(local_path=image_paths, remote_path=remote_folder, overwrite=True)

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
            logger.error(f"Image upload failed: {e}")
            return []

    @staticmethod
    def _run_captioning_model(
        uploaded_items: List[Tuple[dl.Item, dict]], model_id: str, images: List[ImageContent], dataset: dl.Dataset
    ) -> List[str]:
        """Run captioning model using PromptItems created from image items."""
        if not uploaded_items:
            return []

        prompt_items_list = []

        try:
            model = dl.models.get(model_id=model_id)

            if model.status != dl.ModelStatus.DEPLOYED:
                logger.info(f"Model not deployed, deploying now | model_id={model_id}")
                model.deploy()

            # Create PromptItems from image items
            for uploaded_item, metadata in uploaded_items:
                try:
                    # Create a PromptItem from the image item
                    prompt_item = dl.PromptItem.from_item(uploaded_item)

                    # Optionally add a text prompt (some models use default prompt if none provided)
                    # The image is already included in the PromptItem via from_item()
                    prompt_items_list.append((prompt_item, uploaded_item, metadata))

                except Exception as e:
                    logger.warning(f"Failed to create prompt item from image {uploaded_item.id}: {e}")
                    prompt_items_list.append((None, uploaded_item, metadata))

            if not prompt_items_list:
                logger.warning("No prompt items were created")
                return []

            # Filter out None prompt items
            valid_prompt_items = [pi for pi, _, _ in prompt_items_list if pi is not None]
            if not valid_prompt_items:
                logger.warning("No valid prompt items to process")
                return []

            logger.info(f"Running captioning model | model_id={model_id} prompt_items={len(valid_prompt_items)}")

            # Run predictions on the image items (the model adapter will convert them to PromptItems internally)
            # We use the original item IDs since PromptItems are created from items
            item_ids = [uploaded_item.id for _, uploaded_item, _ in prompt_items_list if uploaded_item is not None]
            execution = model.predict(item_ids=item_ids)
            execution.wait()

            if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
                raise Exception(f"Captioning failed: {execution.latest_status.get('message')}")

            # Extract captions from PromptItem messages
            # After execution, we need to recreate PromptItems from the updated items
            descriptions = []
            for prompt_item, uploaded_item, metadata in prompt_items_list:
                if prompt_item is None or uploaded_item is None:
                    descriptions.append("")
                    continue

                try:
                    # Refresh the item and recreate PromptItem to get updated messages
                    updated_item = dl.items.get(item_id=uploaded_item.id)
                    updated_prompt_item = dl.PromptItem.from_item(updated_item)
                    caption = ImageDescriber._extract_caption_from_prompt_item(updated_prompt_item)

                    # Update the corresponding ImageContent object
                    image_idx = metadata['image_index']
                    if image_idx < len(images):
                        images[image_idx].caption = caption

                    descriptions.append(caption)
                except Exception as e:
                    logger.warning(f"Failed to extract caption from prompt item: {e}")
                    descriptions.append("")

            return descriptions

        except Exception as e:
            logger.error(f"Captioning prediction failed: {e}")
            raise

    @staticmethod
    def _extract_caption_from_prompt_item(prompt_item: dl.PromptItem) -> str:
        """Extract caption from PromptItem messages (assistant's response)."""
        caption = ""

        try:
            # Get messages from the PromptItem
            messages = prompt_item.to_messages(model_name=None)

            # Find the assistant's message (the model response)
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    # Extract text content from the assistant's message
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for content_item in content:
                            # Check for text content
                            if isinstance(content_item, dict):
                                mimetype = content_item.get("mimetype")
                                if mimetype == dl.PromptType.TEXT or mimetype == "text":
                                    value = content_item.get("value", "")
                                    if value:
                                        if caption:
                                            caption = f"{caption} {value}".strip()
                                        else:
                                            caption = value
                                # Also check for 'text' key as fallback
                                elif "text" in content_item:
                                    text_value = content_item.get("text", "")
                                    if text_value:
                                        if caption:
                                            caption = f"{caption} {text_value}".strip()
                                        else:
                                            caption = text_value
                    elif isinstance(content, str):
                        caption = content
                    break

            if caption:
                logger.debug(f"Extracted caption from PromptItem | length={len(caption)}")
            else:
                logger.warning("No assistant message found in PromptItem")

        except Exception as e:
            logger.warning(f"Failed to extract caption from PromptItem: {str(e)}")

        return caption.strip() if caption else "No description available"

    @staticmethod
    def _cleanup_resources(
        uploaded_items: List[Tuple[dl.Item, dict]], temp_folder_name: str, dataset: dl.Dataset, temp_local_dir: str
    ) -> None:
        """Clean up temporary resources from captioning."""
        if uploaded_items:
            uploaded_item_objects = [item for item, _ in uploaded_items]
            cleanup_temp_items_and_folder(uploaded_item_objects, temp_folder_name, dataset, temp_local_dir)
        else:
            try:
                shutil.rmtree(temp_local_dir, ignore_errors=True)
            except Exception:
                pass


# Transform wrappers


def ocr_enhance(data: ExtractedData) -> ExtractedData:
    """
    Add OCR text from images to content.

    Routes to appropriate method based on config.ocr_method:
    - 'local': Use EasyOCR locally (default)
    - 'batch': Use Dataloop model for batch processing
    - 'auto': Try batch first, fallback to local on failure
    """
    data.current_stage = "ocr"

    if not data.config.use_ocr:
        return data

    if not data.images:
        return data

    ocr_method = getattr(data.config, 'ocr_method', 'local')

    if ocr_method == 'local':
        return _ocr_local(data)
    elif ocr_method == 'batch':
        return _ocr_batch(data)
    elif ocr_method == 'auto':
        return _ocr_auto(data)
    else:
        logger.warning(f"Unknown ocr_method '{ocr_method}', using local")
        return _ocr_local(data)


def describe_images(data: ExtractedData) -> ExtractedData:
    """Generate image descriptions using Dataloop vision models."""
    data.current_stage = "image_description"

    if not data.images:
        return data

    model_id = data.config.vision_model_id if hasattr(data.config, 'vision_model_id') else None
    if not model_id:
        data.log_warning("Vision model not configured. Skipping image descriptions.")
        return data

    if not data.item:
        data.log_error("Source item not provided. Cannot generate image descriptions.")
        return data

    try:
        descriptions = ImageDescriber.describe(
            images=data.images,
            model_id=model_id,
            dataset=data.item.dataset,
            item_name=data.item.name,
            item_id=data.item.id,
        )

        if descriptions:
            desc_text = '\n\n--- Image Descriptions ---\n\n' + '\n\n'.join(descriptions)
            data.content_text += desc_text

        data.metadata['image_descriptions_generated'] = True
        data.metadata['image_description_count'] = len(descriptions)

    except Exception as e:
        data.log_warning("Image description failed. Check logs for details.")
        logger.exception(f"Image description error: {e}")

    return data


# Private OCR implementations


def _ocr_local(data: ExtractedData) -> ExtractedData:
    """Extract OCR text using local EasyOCR."""
    ocr_by_page = OCREnhancer.extract_local(data.images)

    if not ocr_by_page:
        return data

    return _finalize_ocr(data, ocr_by_page, method='local')


def _ocr_batch(data: ExtractedData) -> ExtractedData:
    """Extract OCR text using Dataloop batch processing."""
    model_id = getattr(data.config, 'ocr_model_id', None)
    if not model_id:
        data.log_warning("OCR model not configured. Using local OCR instead.")
        return _ocr_local(data)

    if not data.item:
        data.log_error("Source item not provided. Cannot perform batch OCR.")
        return _ocr_local(data)

    try:
        ocr_by_page = OCREnhancer.extract_batch(
            images=data.images,
            model_id=model_id,
            dataset=data.item.dataset,
            item_name=data.item.name,
            item_id=data.item.id,
        )

        if not ocr_by_page:
            return data

        return _finalize_ocr(data, ocr_by_page, method='batch')

    except Exception as e:
        data.log_warning("Batch OCR failed. Using local OCR instead.")
        logger.warning(f"Batch OCR error: {e}")
        return _ocr_local(data)


def _ocr_auto(data: ExtractedData) -> ExtractedData:
    """Try batch OCR first, fallback to local on failure."""
    model_id = getattr(data.config, 'ocr_model_id', None)

    if not model_id or not data.item:
        return _ocr_local(data)

    try:
        return _ocr_batch(data)
    except Exception as e:
        logger.warning(f"Batch OCR failed: {e}, falling back to local OCR")
        return _ocr_local(data)


def _finalize_ocr(data: ExtractedData, ocr_by_page: Dict[int, List[str]], method: str) -> ExtractedData:
    """Integrate OCR results into content and set metadata."""
    data.content_text = OCREnhancer.integrate_ocr_per_page(data.content_text, ocr_by_page)

    total_ocr_length = sum(len(t) for texts in ocr_by_page.values() for t in texts)
    total_ocr_count = sum(len(texts) for texts in ocr_by_page.values())

    data.metadata['ocr_applied'] = True
    data.metadata['ocr_method'] = method
    data.metadata['ocr_text_length'] = total_ocr_length
    data.metadata['ocr_image_count'] = total_ocr_count

    return data
