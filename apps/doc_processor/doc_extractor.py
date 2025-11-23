"""
DOCX extraction logic separated from the main processor.

This module handles all DOCX-specific extraction operations:
- Text extraction from paragraphs
- Image extraction from embedded resources
- Table extraction with markdown conversion
"""
import logging
import os
import tempfile
from typing import List, Dict

try:
    from docx import Document
except ImportError:
    Document = None

from utils.extracted_data import ExtractedData
from utils.data_types import ImageContent, TableContent

logger = logging.getLogger(__name__)


class DOCExtractor:
    """
    Handles DOCX extraction operations.

    All methods are static for stateless, concurrent-safe operation.
    """

    @staticmethod
    def extract(data: ExtractedData) -> ExtractedData:
        """
        Extract content from DOCX item.

        Populates:
        - data.content_text: Extracted text from paragraphs
        - data.images: List of ImageContent
        - data.tables: List of TableContent
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

        if Document is None:
            data.log_error("python-docx not installed")
            return data

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = data.item.download(local_path=temp_dir)
                doc = Document(file_path)

                # Extract paragraphs
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                data.content_text = '\n\n'.join(paragraphs)

                # Extract images if configured
                if data.config.extract_images:
                    data.images = DOCExtractor._extract_images(doc, temp_dir)

                # Extract tables if configured
                if data.config.extract_tables:
                    data.tables = DOCExtractor._extract_tables(doc)

                # Set metadata
                data.metadata = {
                    'paragraph_count': len(paragraphs),
                    'source_file': data.item_name,
                    'extraction_method': 'python-docx',
                    'image_count': len(data.images),
                    'table_count': len(data.tables),
                    'processor': 'doc',
                }

        except Exception as e:
            data.log_error(f"DOCX extraction failed: {e}")
            logger.exception("DOCX extraction error")

        return data

    @staticmethod
    def _extract_images(doc, temp_dir: str) -> List[ImageContent]:
        """
        Extract embedded images from DOCX.

        Args:
            doc: python-docx Document object
            temp_dir: Directory to save extracted images

        Returns:
            List of ImageContent objects
        """
        images = []

        try:
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    # Get filename from target reference
                    filename = rel.target_ref.split('/')[-1]
                    image_path = os.path.join(temp_dir, filename)

                    # Write image data
                    with open(image_path, 'wb') as f:
                        f.write(rel.target_part.blob)

                    # Extract extension
                    ext = filename.split('.')[-1] if '.' in filename else None

                    images.append(
                        ImageContent(
                            path=image_path,
                            format=ext,
                            caption=None,
                            page_number=None,
                            bbox=None,
                            size=None,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to extract images from DOCX: {e}")

        return images

    @staticmethod
    def _extract_tables(doc) -> List[TableContent]:
        """
        Extract tables from DOCX with markdown conversion.

        Args:
            doc: python-docx Document object

        Returns:
            List of TableContent objects
        """
        tables = []

        for table in doc.tables:
            try:
                if not table.rows:
                    continue

                # Get headers from first row
                headers = [cell.text.strip() for cell in table.rows[0].cells]

                # Get data rows
                rows = []
                for row in table.rows[1:]:
                    row_data = {}
                    for i, cell in enumerate(row.cells):
                        if i < len(headers):
                            row_data[headers[i]] = cell.text.strip()
                    rows.append(row_data)

                # Convert to markdown
                markdown = DOCExtractor._table_to_markdown(headers, rows)

                tables.append(
                    TableContent(
                        data=rows,
                        markdown=markdown,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to extract table: {e}")

        return tables

    @staticmethod
    def _table_to_markdown(headers: List[str], rows: List[Dict]) -> str:
        """
        Convert table data to markdown format.

        Args:
            headers: List of column headers
            rows: List of row dictionaries

        Returns:
            Markdown-formatted table string
        """
        if not headers:
            return ""

        # Header row
        md = "| " + " | ".join(headers) + " |\n"

        # Separator row
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # Data rows
        for row in rows:
            values = [str(row.get(h, '')) for h in headers]
            md += "| " + " | ".join(values) + " |\n"

        return md
