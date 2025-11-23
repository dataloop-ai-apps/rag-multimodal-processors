"""Tests for PDF and DOC extractors."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile

from utils.extracted_data import ExtractedData
from utils.config import Config
from apps.pdf_processor.pdf_extractor import PDFExtractor
from apps.doc_processor.doc_extractor import DOCExtractor


class TestPDFExtractorBasics:
    """Test PDFExtractor basic behavior."""

    def test_extract_without_item_logs_error(self):
        """Should log error when no item provided."""
        data = ExtractedData()

        result = PDFExtractor.extract(data)

        assert result.errors.has_errors()
        assert "No item provided" in result.errors.errors[0]

    def test_extract_sets_current_stage(self):
        """Should set current_stage to extraction."""
        data = ExtractedData()

        PDFExtractor.extract(data)

        assert data.current_stage == "extraction"

    @patch('apps.pdf_processor.pdf_extractor.fitz', None)
    def test_extract_logs_error_when_fitz_missing(self):
        """Should log error when PyMuPDF not installed."""
        mock_item = Mock()
        mock_item.name = "test.pdf"
        data = ExtractedData(item=mock_item)

        result = PDFExtractor.extract(data)

        assert result.errors.has_errors()
        assert "not installed" in result.errors.errors[0]


class TestPDFExtractorWithMocks:
    """Test PDFExtractor with mocked PDF library."""

    @pytest.fixture
    def mock_fitz(self):
        """Create mock fitz module."""
        with patch('apps.pdf_processor.pdf_extractor.fitz') as mock:
            # Setup mock document
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Page 1 content"
            mock_page.get_images.return_value = []
            mock_doc.__iter__ = lambda self: iter([mock_page])
            mock_doc.__len__ = lambda self: 1
            mock.open.return_value = mock_doc
            yield mock

    @pytest.fixture
    def mock_item(self):
        """Create mock Dataloop item."""
        item = Mock()
        item.name = "test.pdf"
        item.id = "item-123"

        # Mock download to create a temp file
        def mock_download(local_path=None):
            path = os.path.join(local_path or tempfile.gettempdir(), "test.pdf")
            with open(path, 'wb') as f:
                f.write(b'%PDF-1.4 fake pdf content')
            return path

        item.download = mock_download
        return item

    def test_extract_basic_pymupdf(self, mock_fitz, mock_item):
        """Should extract text using basic PyMuPDF."""
        config = Config(extraction_method='basic', extract_images=False)
        data = ExtractedData(item=mock_item, config=config)

        result = PDFExtractor.extract(data)

        assert "Page 1 content" in result.content_text
        assert result.metadata.get('extraction_method') == 'pymupdf'
        assert not result.errors.has_errors()

    def test_extract_populates_metadata(self, mock_fitz, mock_item):
        """Should populate metadata correctly."""
        config = Config(extraction_method='basic', extract_images=False)
        data = ExtractedData(item=mock_item, config=config)

        result = PDFExtractor.extract(data)

        assert result.metadata.get('source_file') == 'test.pdf'
        assert result.metadata.get('processor') == 'pdf'
        assert 'page_count' in result.metadata


class TestDOCExtractorBasics:
    """Test DOCExtractor basic behavior."""

    def test_extract_without_item_logs_error(self):
        """Should log error when no item provided."""
        data = ExtractedData()

        result = DOCExtractor.extract(data)

        assert result.errors.has_errors()
        assert "No item provided" in result.errors.errors[0]

    def test_extract_sets_current_stage(self):
        """Should set current_stage to extraction."""
        data = ExtractedData()

        DOCExtractor.extract(data)

        assert data.current_stage == "extraction"

    @patch('apps.doc_processor.doc_extractor.Document', None)
    def test_extract_logs_error_when_docx_missing(self):
        """Should log error when python-docx not installed."""
        mock_item = Mock()
        mock_item.name = "test.docx"
        data = ExtractedData(item=mock_item)

        result = DOCExtractor.extract(data)

        assert result.errors.has_errors()
        assert "not installed" in result.errors.errors[0]


class TestDOCExtractorWithMocks:
    """Test DOCExtractor with mocked docx library."""

    @pytest.fixture
    def mock_document(self):
        """Create mock Document class."""
        with patch('apps.doc_processor.doc_extractor.Document') as MockDoc:
            mock_doc = MagicMock()

            # Mock paragraphs
            mock_para1 = MagicMock()
            mock_para1.text = "First paragraph"
            mock_para2 = MagicMock()
            mock_para2.text = "Second paragraph"
            mock_doc.paragraphs = [mock_para1, mock_para2]

            # Mock tables (empty)
            mock_doc.tables = []

            # Mock rels for images (empty)
            mock_doc.part.rels.values.return_value = []

            MockDoc.return_value = mock_doc
            yield MockDoc

    @pytest.fixture
    def mock_item(self):
        """Create mock Dataloop item."""
        item = Mock()
        item.name = "test.docx"
        item.id = "item-456"

        def mock_download(local_path=None):
            path = os.path.join(local_path or tempfile.gettempdir(), "test.docx")
            with open(path, 'wb') as f:
                f.write(b'fake docx content')
            return path

        item.download = mock_download
        return item

    def test_extract_paragraphs(self, mock_document, mock_item):
        """Should extract text from paragraphs."""
        config = Config(extract_images=False, extract_tables=False)
        data = ExtractedData(item=mock_item, config=config)

        result = DOCExtractor.extract(data)

        assert "First paragraph" in result.content_text
        assert "Second paragraph" in result.content_text
        assert not result.errors.has_errors()

    def test_extract_populates_metadata(self, mock_document, mock_item):
        """Should populate metadata correctly."""
        config = Config(extract_images=False, extract_tables=False)
        data = ExtractedData(item=mock_item, config=config)

        result = DOCExtractor.extract(data)

        assert result.metadata.get('source_file') == 'test.docx'
        assert result.metadata.get('processor') == 'doc'
        assert result.metadata.get('paragraph_count') == 2


class TestDOCExtractorTableConversion:
    """Test table to markdown conversion."""

    def test_table_to_markdown_basic(self):
        """Should convert table to markdown format."""
        headers = ["Name", "Age", "City"]
        rows = [
            {"Name": "Alice", "Age": "30", "City": "NYC"},
            {"Name": "Bob", "Age": "25", "City": "LA"},
        ]

        result = DOCExtractor._table_to_markdown(headers, rows)

        assert "| Name | Age | City |" in result
        assert "| --- | --- | --- |" in result
        assert "| Alice | 30 | NYC |" in result
        assert "| Bob | 25 | LA |" in result

    def test_table_to_markdown_empty_headers(self):
        """Should handle empty headers."""
        result = DOCExtractor._table_to_markdown([], [])
        assert result == ""

    def test_table_to_markdown_missing_values(self):
        """Should handle missing values in rows."""
        headers = ["A", "B"]
        rows = [{"A": "1"}]  # Missing "B"

        result = DOCExtractor._table_to_markdown(headers, rows)

        assert "| 1 |  |" in result  # Empty value for B


class TestExtractorIntegration:
    """Test extractor integration with ExtractedData."""

    def test_pdf_extractor_error_tracking(self):
        """PDFExtractor should integrate with error tracking."""
        data = ExtractedData(config=Config(error_mode='continue', max_errors=5))

        PDFExtractor.extract(data)  # Will fail - no item

        assert data.errors.has_errors()
        assert data.current_stage == "extraction"

    def test_doc_extractor_error_tracking(self):
        """DOCExtractor should integrate with error tracking."""
        data = ExtractedData(config=Config(error_mode='continue', max_errors=5))

        DOCExtractor.extract(data)  # Will fail - no item

        assert data.errors.has_errors()
        assert data.current_stage == "extraction"

    def test_extractor_respects_config(self):
        """Extractors should respect configuration."""
        config = Config(extract_images=False, extract_tables=False)
        data = ExtractedData(config=config)

        # Verify config is accessible
        assert data.config.extract_images is False
        assert data.config.extract_tables is False
