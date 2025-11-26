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
from typing import List

try:
    import fitz
    import pymupdf4llm
    import pymupdf.layout
except ImportError:
    fitz = None
    pymupdf4llm = None

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

        if fitz is None:
            data.log_error("Required PDF library not installed. Check logs for details.")
            logger.error("PyMuPDF (fitz) is required for PDF extraction")
            return data

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = data.item.download(local_path=temp_dir)
                use_markdown = data.config.extraction_method == 'markdown'

                if use_markdown and pymupdf4llm is not None:
                    PDFExtractor._extract_markdown(data, file_path, temp_dir)
                else:
                    PDFExtractor._extract_pymupdf(data, file_path, temp_dir)

        except Exception as e:
            data.log_error("PDF extraction failed. Check logs for details.")
            logger.exception(f"PDF extraction error: {e}")

        return data

    @staticmethod
    def _extract_pymupdf(data: ExtractedData, file_path: str, temp_dir: str) -> None:
        """Extract using basic PyMuPDF."""
        doc = fitz.open(file_path)
        text_parts = []
        images = []

        try:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")

                if data.config.extract_images:
                    page_images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    images.extend(page_images)

            data.content_text = ''.join(text_parts)
            data.images = images
            data.metadata = {
                'page_count': len(doc),
                'source_file': data.item_name,
                'extraction_method': 'pymupdf',
                'image_count': len(images),
                'processor': 'pdf',
            }
        finally:
            doc.close()

    @staticmethod
    def _extract_markdown(data: ExtractedData, file_path: str, temp_dir: str) -> None:
        """Extract using pymupdf4llm with ML-based layout enhancement."""
        data.content_text = pymupdf4llm.to_markdown(file_path)
        images = []

        if data.config.extract_images:
            doc = fitz.open(file_path)
            try:
                for page_num, page in enumerate(doc):
                    page_images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    images.extend(page_images)
            finally:
                doc.close()

        data.images = images
        data.metadata = {
            'source_file': data.item_name,
            'extraction_method': 'pymupdf4llm',
            'format': 'markdown',
            'layout_enhancement': True,
            'image_count': len(images),
            'processor': 'pdf',
        }

    @staticmethod
    def _extract_images_from_page(page, page_num: int, temp_dir: str) -> List[ImageContent]:
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
