"""
Refactored PDF processor using the new pipeline architecture.
"""

import tempfile
import fitz
import pymupdf4llm
import os
from typing import List, Dict, Any
import dtlpy as dl
from pipeline.base.processor import BaseProcessor, ProcessorError
from pipeline.utils.logging_utils import ProcessorLogger, ErrorHandler, FileValidator
from extractors.ocr_extractor import OCRExtractor


class PDFProcessor(BaseProcessor):
    """
    Refactored PDF processor using the new pipeline architecture.
    """

    def __init__(self):
        """Initialize PDF processor."""
        super().__init__('pdf')
        self.logger = ProcessorLogger('pdf')
        self.error_handler = ErrorHandler('pdf')
        self.validator = FileValidator()

        # Download required NLTK data (only if not already present)
        import nltk

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            self.logger.info("Downloading NLTK averaged_perceptron_tagger")
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def _extract_content(self, item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from PDF file.

        Args:
            item: PDF file item
            config: Processing configuration

        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Download file to temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)

                # Validate file
                if not self.validator.validate_file_exists(file_path):
                    raise ProcessorError(f"File not found: {file_path}")

                if not self.validator.validate_file_size(file_path, max_size_mb=100):
                    raise ProcessorError(f"File too large: {file_path}")

                # Extract PDF content
                content, metadata = self._extract_pdf_content(file_path, item, config)

                self.logger.info(
                    f"Extracted PDF content",
                    file_path=file_path,
                    content_length=len(content),
                    total_pages=metadata.get('total_pages', 0),
                )

                return {'content': content, 'metadata': metadata}

        except Exception as e:
            error_msg = self.error_handler.handle_file_error(item.name, e)
            raise ProcessorError(error_msg)

    def _extract_pdf_content(self, file_path: str, item: dl.Item, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Extract content from PDF file.

        Args:
            file_path: Path to the PDF file
            item: Original PDF item
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        # Extract configuration parameters
        ocr_from_images = config.get('ocr_from_images', False)
        ocr_integration_method = config.get('ocr_integration_method', 'append_to_page')
        use_markdown_extraction = config.get('use_markdown_extraction', False)

        self.logger.info(
            f"PDF extraction config | markdown={use_markdown_extraction} "
            f"ocr_from_images={ocr_from_images} ocr_method={ocr_integration_method}"
        )

        # Extract text content
        if use_markdown_extraction:
            page_texts, total_pages = self._extract_text_as_markdown(file_path, item.id)
        else:
            page_texts, total_pages = self._extract_text(file_path, item.id)

        # Extract OCR content if requested
        ocr_texts = []
        if ocr_from_images:
            self.logger.info("OCR enabled | using EasyOCR")
            ocr_extractor = OCRExtractor(model_id=None)
            ocr_texts = self._extract_and_ocr_with_easyocr(file_path, item.id, ocr_extractor)

        # Combine text and OCR content
        combined_text = self._combine_texts(page_texts, ocr_texts, ocr_integration_method)

        # Get processor-specific metadata
        metadata = self._get_pdf_metadata(total_pages, use_markdown_extraction, config)

        return combined_text, metadata

    def _extract_text(self, pdf_path: str, item_id: str) -> tuple[List[str], int]:
        """
        Extract text from PDF using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            item_id: Item ID for logging

        Returns:
            Tuple of (page_texts, total_pages)
        """
        self.logger.info(f"Extracting text | item_id={item_id}")

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            page_texts = []
            for page in doc:
                page_texts.append(page.get_text())

            self.logger.info(f"Extracted {len(page_texts)} pages")
            return page_texts, total_pages

    def _extract_text_as_markdown(self, pdf_path: str, item_id: str) -> tuple[List[str], int]:
        """
        Extract text as markdown using pymupdf4llm.

        Args:
            pdf_path: Path to PDF file
            item_id: Item ID for logging

        Returns:
            Tuple of (page_texts, total_pages)
        """
        self.logger.info(f"Extracting markdown | item_id={item_id}")

        try:
            md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=False, show_progress=True)

            if isinstance(md_text, list):
                page_texts = []
                for page_data in md_text:
                    if isinstance(page_data, dict):
                        page_texts.append(page_data.get('text', ''))
                    else:
                        page_texts.append(str(page_data))
            elif isinstance(md_text, str):
                page_texts = md_text.split('\n-----\n')
            else:
                self.logger.warning("Unexpected markdown format, falling back")
                return self._extract_text(pdf_path, item_id)

            total_pages = len(page_texts)
            self.logger.info(f"Extracted {total_pages} pages as markdown")
            return page_texts, total_pages

        except Exception as e:
            self.logger.error(f"Markdown extraction failed: {e}")
            return self._extract_text(pdf_path, item_id)

    def _extract_and_ocr_with_easyocr(
        self, pdf_path: str, item_id: str, ocr_extractor: OCRExtractor
    ) -> List[Dict[str, Any]]:
        """
        Extract images from PDF and apply OCR using EasyOCR.

        Args:
            pdf_path: Path to PDF file
            item_id: Item ID for logging
            ocr_extractor: OCR extractor instance

        Returns:
            List of OCR results
        """
        self.logger.info(f"Extracting images for EasyOCR | item_id={item_id}")

        images = self._extract_images_from_pdf(pdf_path)
        return self._process_images_with_easyocr(images, item_id, ocr_extractor)

    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all images from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of image dictionaries
        """
        images = []

        with fitz.open(pdf_path) as pdf_file:
            for page_index in range(len(pdf_file)):
                page = pdf_file.load_page(page_index)
                image_list = page.get_images(full=True)

                for image_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        images.append(
                            {
                                'bytes': base_image["image"],
                                'page_index': page_index,
                                'image_index': image_index,
                                'extension': base_image.get("ext", "png"),
                            }
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image page={page_index} img={image_index}: {e}")
                        continue

        self.logger.info(f"Extracted {len(images)} images from PDF")
        return images

    def _process_images_with_easyocr(
        self, images: List[Dict[str, Any]], item_id: str, ocr_extractor: OCRExtractor
    ) -> List[Dict[str, Any]]:
        """
        Process images with EasyOCR.

        Args:
            images: List of image dictionaries
            item_id: Item ID for logging
            ocr_extractor: OCR extractor instance

        Returns:
            List of OCR results
        """
        if not images:
            self.logger.info("No images to process")
            return []

        self.logger.info(f"Processing {len(images)} images with EasyOCR | item_id={item_id}")

        ocr_results = []
        for image_data in images:
            try:
                # Save to temp file for EasyOCR processing
                temp_image_path = os.path.join(
                    tempfile.gettempdir(),
                    f"ocr_img_{item_id}_{image_data['page_index']}_{image_data['image_index']}.{image_data['extension']}",
                )

                with open(temp_image_path, 'wb') as f:
                    f.write(image_data['bytes'])

                # Run EasyOCR on local file
                ocr_text = ocr_extractor.extract_text(temp_image_path)

                # Clean up temp file
                try:
                    os.remove(temp_image_path)
                except:
                    pass

                ocr_results.append(
                    {
                        'page_index': image_data['page_index'],
                        'image_index': image_data['image_index'],
                        'text': ocr_text,
                        'extension': image_data['extension'],
                    }
                )
            except Exception as e:
                self.logger.warning(
                    f"EasyOCR failed on image page={image_data['page_index']} img={image_data['image_index']}: {e}"
                )
                continue

        self.logger.info(f"EasyOCR completed | images_processed={len(ocr_results)}")
        return ocr_results

    def _combine_texts(self, page_texts: List[str], ocr_texts: List[Dict[str, Any]], integration_method: str) -> str:
        """
        Combine PDF text and OCR text.

        Args:
            page_texts: List of page texts
            ocr_texts: List of OCR results
            integration_method: Method to integrate OCR text

        Returns:
            Combined text
        """
        if integration_method == 'append_to_page':
            combined_pages = page_texts.copy()
            for ocr_result in ocr_texts:
                page_idx = ocr_result['page_index']
                if page_idx < len(combined_pages):
                    combined_pages[page_idx] += (
                        f"\n\n[OCR_IMAGE_{ocr_result['image_index']}]\n{ocr_result['text']}"
                        if ocr_result['text']
                        else ''
                    )
            return '\n\n'.join(combined_pages)

        elif integration_method == 'separate_chunks':
            pdf_text = '\n\n'.join(page_texts)
            ocr_text = '\n\n'.join(
                [
                    f"[OCR_PAGE_{r['page_index']}_IMAGE_{r['image_index']}]\n{r['text']}" if r['text'] else ''
                    for r in ocr_texts
                ]
            )
            return f"{pdf_text}\n\n[OCR_SECTION]\n{ocr_text}"

        else:  # combine_all
            all_text = '\n\n'.join(page_texts)
            for ocr_result in ocr_texts:
                all_text += (
                    f"\n\n[OCR_PAGE_{ocr_result['page_index']}_IMAGE_{ocr_result['image_index']}]\n{ocr_result['text']}"
                    if ocr_result['text']
                    else ''
                )
            return all_text

    def _get_pdf_metadata(
        self, total_pages: int, use_markdown_extraction: bool, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get PDF-specific metadata.

        Args:
            total_pages: Total number of pages
            use_markdown_extraction: Whether markdown extraction was used
            config: Processing configuration

        Returns:
            Dictionary with PDF metadata
        """
        metadata = {
            'file_type': 'pdf',
            'total_pages': total_pages,
            'extraction_method': 'pymupdf4llm' if use_markdown_extraction else 'fitz',
            'extraction_format': 'markdown' if use_markdown_extraction else 'plain',
            'markdown_aware_splitting': use_markdown_extraction,
            'ocr_from_images': config.get('ocr_from_images', False),
            'ocr_integration_method': config.get('ocr_integration_method', 'append_to_page'),
        }
        return metadata

    def _get_processor_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processor-specific metadata.

        Args:
            config: Processing configuration

        Returns:
            Dictionary with processor-specific metadata
        """
        metadata = super()._get_processor_metadata(config)
        metadata.update(
            {
                'processor_type': 'pdf',
                'ocr_from_images': config.get('ocr_from_images', False),
                'use_markdown_extraction': config.get('use_markdown_extraction', False),
                'ocr_integration_method': config.get('ocr_integration_method', 'append_to_page'),
            }
        )
        return metadata


