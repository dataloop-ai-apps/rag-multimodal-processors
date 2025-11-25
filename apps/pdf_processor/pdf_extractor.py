"""
PDF extraction logic separated from the main processor.

This module handles all PDF-specific extraction operations:
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
    import pymupdf.layout  # Activates ML-based layout enhancement
except ImportError:
    fitz = None
    pymupdf4llm = None

from utils.extracted_data import ExtractedData
from utils.data_types import ImageContent

logger = logging.getLogger("rag-preprocessor")


class PDFExtractor:
    """
    Handles PDF extraction operations.

    All methods are static for stateless, concurrent-safe operation.
    """

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """
        Extract content from PDF item.

        Populates:
        - data.content_text: Extracted text
        - data.images: List of ImageContent
        - data.metadata: Document metadata

        Args:
            data: ExtractedData with item set

        Returns:
            ExtractedData with extraction results
        """
        data.current_stage = "extraction"

        if not data.item:
            data.log_error("No item provided for extraction")
            return data

        if fitz is None:
            data.log_error("PyMuPDF (fitz) not installed")
            return data

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = data.item.download(local_path=temp_dir)

                # Choose extraction method
                use_markdown = data.config.extraction_method == 'markdown'

                if use_markdown and pymupdf4llm is not None:
                    PDFExtractor._extract_with_markdown(data, file_path, temp_dir)
                else:
                    PDFExtractor._extract_with_pymupdf(data, file_path, temp_dir)

        except Exception as e:
            data.log_error(f"PDF extraction failed: {e}")
            logger.exception("PDF extraction error")

        return data

    @staticmethod
    def _extract_with_pymupdf(data: ExtractedData, file_path: str, temp_dir: str) -> None:
        """
        Extract using basic PyMuPDF.

        Args:
            data: ExtractedData to populate
            file_path: Path to downloaded PDF
            temp_dir: Temporary directory for images
        """
        doc = fitz.open(file_path)
        text_parts = []

        try:
            for page_num, page in enumerate(doc):
                # Extract text
                page_text = page.get_text()
                text_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")

                # Extract images if configured
                if data.config.extract_images:
                    images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    data.images.extend(images)

            data.content_text = ''.join(text_parts)
            data.metadata = {
                'page_count': len(doc),
                'source_file': data.item_name,
                'extraction_method': 'pymupdf',
                'image_count': len(data.images),
                'processor': 'pdf',
            }
        finally:
            doc.close()

    @staticmethod
    def _extract_with_markdown(data: ExtractedData, file_path: str, temp_dir: str) -> None:
        """
        Extract using pymupdf4llm with ML-based layout enhancement.

        Args:
            data: ExtractedData to populate
            file_path: Path to downloaded PDF
            temp_dir: Temporary directory for images
        """
        # Extract markdown text
        md_text = pymupdf4llm.to_markdown(file_path)
        data.content_text = md_text

        # Extract images separately if configured
        if data.config.extract_images:
            doc = fitz.open(file_path)
            try:
                for page_num, page in enumerate(doc):
                    images = PDFExtractor._extract_images_from_page(page, page_num, temp_dir)
                    data.images.extend(images)
            finally:
                doc.close()

        data.metadata = {
            'source_file': data.item_name,
            'extraction_method': 'pymupdf4llm',
            'format': 'markdown',
            'layout_enhancement': True,
            'image_count': len(data.images),
            'processor': 'pdf',
        }

    @staticmethod
    def _extract_images_from_page(page, page_num: int, temp_dir: str) -> List[ImageContent]:
        """
        Extract images from a PDF page with positional metadata.

        Args:
            page: PyMuPDF page object
            page_num: Zero-based page number
            temp_dir: Directory to save extracted images

        Returns:
            List of ImageContent objects
        """
        images = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                # Save image to temp file
                ext = base_image.get('ext', 'png')
                image_path = os.path.join(temp_dir, f"page{page_num}_img{img_index}.{ext}")
                with open(image_path, 'wb') as f:
                    f.write(base_image['image'])

                # Get bounding box
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
