"""
PDF extraction logic.

Handles PDF-specific extraction operations:
- Text extraction (basic PyMuPDF or markdown with ML layout)
- Image extraction with positional metadata
- Metadata collection
"""

import logging
import os
import tempfile
from typing import List, Tuple, Dict, Any

import fitz
import pymupdf.layout
import pymupdf4llm

from utils.extracted_data import ExtractedData
from utils.data_types import ImageContent

logger = logging.getLogger("rag-preprocessor")


class PDFExtractor:
    """PDF extraction operations."""

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """Extract content from PDF item."""
        data.current_stage = "extraction"

        if not data.item:
            data.log_error("No item provided for extraction")
            return data

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = data.item.download(local_path=temp_dir)
                use_markdown = data.config.use_markdown_extraction
                extract_images = data.config.extract_images

                if use_markdown:
                    content, images, metadata = PDFExtractor._extract_markdown(file_path, temp_dir, extract_images)
                else:
                    content, images, metadata = PDFExtractor._extract_pymupdf(file_path, temp_dir, extract_images)

                data.content_text = content
                data.images = images
                metadata['source_file'] = data.item_name
                data.metadata = metadata

        except Exception as e:
            data.log_error("PDF extraction failed. Check logs for details.")
            logger.exception(f"PDF extraction error: {e}")

        return data

    @staticmethod
    def _extract_pymupdf(
        file_path: str, temp_dir: str, extract_images: bool
    ) -> Tuple[str, List[ImageContent], Dict[str, Any]]:
        """Extract text and images using basic PyMuPDF."""
        doc = fitz.open(file_path)
        text_parts = []
        images = []

        try:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")

                if extract_images:
                    page_images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    images.extend(page_images)

            metadata = {
                'page_count': len(doc),
                'extraction_method': 'pymupdf',
                'image_count': len(images),
                'processor': 'pdf',
            }
        finally:
            doc.close()

        return ''.join(text_parts), images, metadata

    @staticmethod
    def _extract_markdown(
        file_path: str, temp_dir: str, extract_images: bool
    ) -> Tuple[str, List[ImageContent], Dict[str, Any]]:
        """Extract text and images using pymupdf4llm with ML-based layout."""
        content = pymupdf4llm.to_markdown(file_path)
        images = []

        if extract_images:
            doc = fitz.open(file_path)
            try:
                for page_num, page in enumerate(doc):
                    page_images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    images.extend(page_images)
            finally:
                doc.close()

        metadata = {
            'extraction_method': 'pymupdf4llm',
            'format': 'markdown',
            'layout_enhancement': True,
            'image_count': len(images),
            'processor': 'pdf',
        }

        return content, images, metadata

    @staticmethod
    def _extract_images_from_page(page: fitz.Page, page_num: int, temp_dir: str) -> List[ImageContent]:
        """Extract images from a PDF page with positional metadata."""
        images = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                ext = base_image.get('ext', 'png')
                image_path = os.path.join(temp_dir, f"page{page_num}_img{img_index}.{ext}")
                with open(image_path, 'wb') as f:
                    f.write(base_image['image'])

                bbox = None
                image_rects = page.get_image_rects(xref)
                if image_rects:
                    rect = image_rects[0] if isinstance(image_rects, list) else image_rects
                    bbox = (rect.x0, rect.y0, rect.width, rect.height)

                images.append(
                    ImageContent(
                        path=image_path,
                        page_number=page_num + 1,
                        format=ext,
                        size=(base_image.get('width'), base_image.get('height')),
                        bbox=bbox,
                    )
                )
            except (IOError, OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")

        return images
