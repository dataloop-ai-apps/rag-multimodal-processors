"""
PDF extractor for extracting text, images, and tables from PDF files.
"""

import os
import fitz
import tempfile
from fitz import Document
import pymupdf4llm
from typing import List, Dict, Any
import dtlpy as dl
from .content_types import ExtractedContent, ImageContent


class PDFExtractor:
    """Extract text, images, and tables from PDF files"""

    def __init__(self):
        self.mime_type = 'application/pdf'
        self.name = 'PDF'

    def extract(self, item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """Extract all content from PDF"""

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)

            # Choose extraction method
            if config.get('use_markdown_extraction', False):
                return self._extract_with_markdown(file_path, item, temp_dir, config)
            else:
                return self._extract_with_pymupdf(file_path, item, temp_dir, config)

    def _extract_with_pymupdf(
        self, file_path: str, item: dl.Item, temp_dir: str, config: Dict[str, Any]
    ) -> ExtractedContent:
        """Extract using basic PyMuPDF"""

        doc = fitz.open(file_path)
        result = ExtractedContent()
        text_parts = []

        for page_num, page in enumerate(doc):
            # Extract text
            page_text = page.get_text()
            text_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")

            # Extract images if requested
            if config.get('extract_images', True):
                images = self._extract_images_from_page(page, page_num, temp_dir)
                result.images.extend(images)

        result.text = ''.join(text_parts)
        result.metadata = {
            'page_count': len(doc),
            'source_file': item.name,
            'extraction_method': 'pymupdf',
            'image_count': len(result.images),
            'table_count': len(result.tables),
            'extractor': 'pdf',
        }

        doc.close()
        return result

    def _extract_with_markdown(
        self, file_path: str, item: dl.Item, temp_dir: str, config: Dict[str, Any]
    ) -> ExtractedContent:
        """Extract using pymupdf4llm (preserves structure)"""

        # Extract as markdown
        md_text = pymupdf4llm.to_markdown(file_path)

        result = ExtractedContent()
        result.text = md_text

        # Still extract images if requested
        if config.get('extract_images', True):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                images = self._extract_images_from_page(page, page_num, temp_dir)
                result.images.extend(images)
            doc.close()

        result.metadata = {
            'source_file': item.name,
            'extraction_method': 'pymupdf4llm',
            'format': 'markdown',
            'image_count': len(result.images),
            'extractor': 'pdf',
        }

        return result

    def _extract_images_from_page(self, page, page_num: int, temp_dir: str) -> List[ImageContent]:
        """
        Extract images from a PDF page with positional metadata.

        Extracts images along with their bounding box positions on the page.
        """
        images = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                image_path = os.path.join(temp_dir, f"page{page_num}_img{img_index}.{base_image['ext']}")
                with open(image_path, 'wb') as f:
                    f.write(base_image['image'])

                # Get image bounding boxes/positions on the page
                bbox = None
                image_rects = page.get_image_rects(xref)
                if image_rects:
                    # Use the first (or largest) rectangle if multiple found
                    rect = image_rects[0] if isinstance(image_rects, list) else image_rects
                    # Convert fitz.Rect to (x0, y0, x1, y1) then to (x, y, width, height)
                    bbox = (rect.x0, rect.y0, rect.width, rect.height)  # x position  # y position  # width  # height

                images.append(
                    ImageContent(
                        path=image_path,
                        page_number=page_num + 1,
                        format=base_image['ext'],
                        size=(base_image.get('width'), base_image.get('height')),
                        bbox=bbox,
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to extract image {img_index} from page {page_num}: {e}")

        return images
