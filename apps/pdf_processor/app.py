"""
PDF processor app.

Self-contained PDF processor with all extraction and processing logic.
"""

import logging
import os
import tempfile
from typing import Dict, Any, List

import dtlpy as dl
import fitz
import nltk
import pymupdf.layout  # Activates ML-based layout enhancement in pymupdf4llm
import pymupdf4llm

import transforms
from utils.data_types import ExtractedContent, ImageContent

logger = logging.getLogger("rag-preprocessor")


class PDFProcessor(dl.BaseServiceRunner):
    """
    Unified PDF Processor for extracting text, applying OCR, and creating chunks.

    Supports:
    - Text extraction (plain and markdown-aware)
    - Image extraction and OCR
    - Multiple chunking strategies
    - Text cleaning and normalization
    """

    def __init__(self):
        """Initialize PDF processor."""
        # Configure Dataloop client timeouts
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30

        # Download required NLTK data (only if not already present)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

    @staticmethod
    def extract_pdf(item: dl.Item, config: Dict[str, Any]) -> ExtractedContent:
        """
        Extract text and images from PDF (formerly in PDFExtractor).

        Args:
            item: Dataloop PDF item to extract from
            config: Configuration dict

        Returns:
            ExtractedContent: Extracted content with text, images, and metadata
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = item.download(local_path=temp_dir)

            if config.get('use_markdown_extraction', False):
                return PDFProcessor._extract_with_markdown(file_path, item, temp_dir, config)
            else:
                return PDFProcessor._extract_with_pymupdf(file_path, item, temp_dir, config)

    @staticmethod
    def _extract_with_pymupdf(file_path: str, item: dl.Item, temp_dir: str, config: Dict[str, Any]) -> ExtractedContent:
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
                images = PDFProcessor._extract_images_from_page(page, page_num, temp_dir)
                result.images.extend(images)

        result.text = ''.join(text_parts)
        result.metadata = {
            'page_count': len(doc),
            'source_file': item.name,
            'extraction_method': 'pymupdf',
            'image_count': len(result.images),
            'table_count': len(result.tables),
            'processor': 'pdf',
        }

        doc.close()
        return result

    @staticmethod
    def _extract_with_markdown(
        file_path: str, item: dl.Item, temp_dir: str, config: Dict[str, Any]
    ) -> ExtractedContent:
        """
        Extract using pymupdf4llm with ML-based layout enhancement.

        Automatically enhances extraction with:
        - ML-based layout analysis
        - Automatic OCR evaluation
        - Better header/footer detection
        """
        md_text = pymupdf4llm.to_markdown(file_path)

        result = ExtractedContent()
        result.text = md_text

        if config.get('extract_images', True):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                images = PDFProcessor._extract_images_from_page(page, page_num, temp_dir)
                result.images.extend(images)
            doc.close()

        result.metadata = {
            'source_file': item.name,
            'extraction_method': 'pymupdf4llm_layout',
            'format': 'markdown',
            'layout_enhancement': True,
            'image_count': len(result.images),
            'processor': 'pdf',
        }

        return result

    @staticmethod
    def _extract_images_from_page(page, page_num: int, temp_dir: str) -> List[ImageContent]:
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
                    bbox = (rect.x0, rect.y0, rect.width, rect.height)

                images.append(
                    ImageContent(
                        path=image_path,
                        page_number=page_num + 1,
                        format=base_image['ext'],
                        size=(base_image.get('width'), base_image.get('height')),
                        bbox=bbox,
                    )
                )
            except (IOError, OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")

        return images

    @staticmethod
    def extract(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from PDF file."""
        item = data.get('item')
        if not item:
            raise ValueError("Missing 'item' in data")

        extracted = PDFProcessor.extract_pdf(item, config)
        data.update(extracted.to_dict())
        return data

    @staticmethod
    def apply_ocr(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply OCR if enabled in config."""
        ocr_enabled = config.get('ocr_from_images', False) or config.get('use_ocr', False)
        if not ocr_enabled:
            return data

        ocr_config = config.copy()
        ocr_config['use_ocr'] = True
        # Map integration method values
        integration_method = ocr_config.get('ocr_integration_method', 'append_to_page')
        method_mapping = {'append_to_page': 'per_page', 'separate_chunks': 'separate', 'combine_all': 'append'}
        ocr_config['ocr_integration_method'] = method_mapping.get(integration_method, integration_method)
        return transforms.ocr_enhance(data, ocr_config)

    @staticmethod
    def clean(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        data = transforms.clean_text(data, config)
        data = transforms.normalize_whitespace(data, config)
        return data

    @staticmethod
    def chunk(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        strategy = config.get('chunking_strategy', 'recursive')
        link_images = config.get('link_images_to_chunks', True)
        embed_images = config.get('embed_images_in_chunks', False)
        has_images = len(data.get('images', [])) > 0

        if strategy == 'recursive' and has_images:
            if embed_images:
                data = transforms.chunk_with_embedded_images(data, config)
            elif link_images:
                data = transforms.chunk_recursive_with_images(data, config)
            else:
                data = transforms.chunk_text(data, config)
        elif strategy == 'semantic':
            data = transforms.llm_chunk_semantic(data, config)
        else:
            data = transforms.chunk_text(data, config)
        return data

    @staticmethod
    def upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks to Dataloop."""
        return transforms.upload_to_dataloop(data, config)

    @staticmethod
    def process_document(item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Dataloop pipeline entry point."""
        config = context.node.metadata.get('customNodeConfig', {})
        return PDFProcessor.run(item, target_dataset, config)

    @staticmethod
    def run(item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any]) -> List[dl.Item]:
        """
        Process a PDF document into chunks.

        Args:
            item: PDF item to process
            target_dataset: Target dataset for storing chunks
            config: Processing configuration dict

        Returns:
            List of uploaded chunk items
        """
        try:
            data = {'item': item, 'target_dataset': target_dataset}
            data = PDFProcessor.extract(data, config)
            data = PDFProcessor.apply_ocr(data, config)
            data = PDFProcessor.clean(data, config)
            data = PDFProcessor.chunk(data, config)
            data = PDFProcessor.upload(data, config)

            uploaded = data.get('uploaded_items', [])
            logger.info(f"Processed {item.name}: {len(uploaded)} chunks")
            return uploaded

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
