"""
Unit tests for all processors.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pipeline.tests.test_framework import ProcessorTestCase, TestDataGenerator
from apps.text_processor.text_processor import TextProcessor
from apps.html_processor.html_processor import HTMLProcessor
from apps.email_processor.email_processor import EmailProcessor
from apps.pdf_processor.pdf_processor_new import PDFProcessor


class TextProcessorTest(ProcessorTestCase):
    """Test cases for TextProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = TextProcessor()

    def test_process_txt_file(self):
        """Test processing a text file."""
        # Create test file
        content = TestDataGenerator.generate_text_content()
        file_path = self.create_test_file(content, "test.txt")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_csv_file(self):
        """Test processing a CSV file."""
        # Create test CSV file
        content = TestDataGenerator.generate_csv_content()
        file_path = self.create_test_file(content, "test.csv")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        # Create test markdown file
        content = "# Test Document\n\nThis is a **markdown** file."
        file_path = self.create_test_file(content, "test.md")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_nonexistent_file(self):
        """Test processing a nonexistent file."""
        # Mock item download to return nonexistent file
        with patch.object(self.mock_item, 'download', return_value="/nonexistent/file.txt"):
            with self.assertRaises(Exception):
                self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)


class HTMLProcessorTest(ProcessorTestCase):
    """Test cases for HTMLProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = HTMLProcessor()

    def test_process_html_file(self):
        """Test processing an HTML file."""
        # Create test HTML file
        content = TestDataGenerator.generate_html_content()
        file_path = self.create_test_file(content, "test.html")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_html_with_beautifulsoup(self):
        """Test processing HTML with BeautifulSoup."""
        # Create test HTML file
        content = TestDataGenerator.generate_html_content()
        file_path = self.create_test_file(content, "test.html")

        # Mock BeautifulSoup
        with patch('apps.html_processor.html_processor.BeautifulSoup') as mock_bs:
            mock_soup = Mock()
            mock_soup.find.return_value = Mock(get_text=lambda: "Test Document")
            mock_soup.find_all.return_value = []
            mock_bs.return_value = mock_soup

            # Mock item download
            with patch.object(self.mock_item, 'download', return_value=file_path):
                result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_html_without_beautifulsoup(self):
        """Test processing HTML without BeautifulSoup."""
        # Create test HTML file
        content = TestDataGenerator.generate_html_content()
        file_path = self.create_test_file(content, "test.html")

        # Mock BeautifulSoup import to fail
        with patch('apps.html_processor.html_processor.BeautifulSoup', side_effect=ImportError):
            # Mock item download
            with patch.object(self.mock_item, 'download', return_value=file_path):
                result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)


class EmailProcessorTest(ProcessorTestCase):
    """Test cases for EmailProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = EmailProcessor()

    def test_process_email_file(self):
        """Test processing an email file."""
        # Create test email file
        content = TestDataGenerator.generate_email_content()
        file_path = self.create_test_file(content, "test.eml")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_multipart_email(self):
        """Test processing a multipart email."""
        # Create test multipart email
        content = """From: sender@example.com
To: recipient@example.com
Subject: Test Multipart Email
Date: Mon, 01 Jan 2024 12:00:00 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=utf-8

This is the plain text version.

--boundary123
Content-Type: text/html; charset=utf-8

<html><body><p>This is the HTML version.</p></body></html>

--boundary123--"""

        file_path = self.create_test_file(content, "test_multipart.eml")

        # Mock item download
        with patch.object(self.mock_item, 'download', return_value=file_path):
            result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)


class PDFProcessorTest(ProcessorTestCase):
    """Test cases for PDFProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = PDFProcessor()

    def test_process_pdf_file(self):
        """Test processing a PDF file."""
        # Create test PDF file
        content = TestDataGenerator.generate_pdf_content()
        file_path = os.path.join(self.temp_dir, "test.pdf")
        with open(file_path, 'wb') as f:
            f.write(content)

        # Mock PyMuPDF
        with patch('apps.pdf_processor.pdf_processor_new.fitz') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__.return_value = mock_doc
            mock_doc.__exit__.return_value = None

            mock_page = Mock()
            mock_page.get_text.return_value = "Sample PDF text"
            mock_doc.__iter__.return_value = [mock_page]

            mock_fitz.open.return_value = mock_doc

            # Mock item download
            with patch.object(self.mock_item, 'download', return_value=file_path):
                result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_pdf_with_markdown_extraction(self):
        """Test processing PDF with markdown extraction."""
        # Create test PDF file
        content = TestDataGenerator.generate_pdf_content()
        file_path = os.path.join(self.temp_dir, "test.pdf")
        with open(file_path, 'wb') as f:
            f.write(content)

        # Mock pymupdf4llm
        with patch('apps.pdf_processor.pdf_processor_new.pymupdf4llm') as mock_pymupdf4llm:
            mock_pymupdf4llm.to_markdown.return_value = [{"text": "Sample markdown text"}]

            # Mock item download
            with patch.object(self.mock_item, 'download', return_value=file_path):
                # Update config to use markdown extraction
                self.mock_context.node.metadata['customNodeConfig']['use_markdown_extraction'] = True

                result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)

    def test_process_pdf_with_ocr(self):
        """Test processing PDF with OCR."""
        # Create test PDF file
        content = TestDataGenerator.generate_pdf_content()
        file_path = os.path.join(self.temp_dir, "test.pdf")
        with open(file_path, 'wb') as f:
            f.write(content)

        # Mock PyMuPDF
        with patch('apps.pdf_processor.pdf_processor_new.fitz') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__.return_value = mock_doc
            mock_doc.__exit__.return_value = None

            mock_page = Mock()
            mock_page.get_text.return_value = "Sample PDF text"
            mock_page.get_images.return_value = []
            mock_doc.__iter__.return_value = [mock_page]

            mock_fitz.open.return_value = mock_doc

        # Mock OCR extractor
        with patch('apps.pdf_processor.pdf_processor_new.OCRExtractor') as mock_ocr:
            mock_ocr_instance = Mock()
            mock_ocr_instance.extract_text.return_value = "OCR text"
            mock_ocr.return_value = mock_ocr_instance

            # Mock item download
            with patch.object(self.mock_item, 'download', return_value=file_path):
                # Update config to enable OCR
                self.mock_context.node.metadata['customNodeConfig']['ocr_from_images'] = True

                result = self.processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        self.assert_processing_success(result)


class ProcessorIntegrationTest(ProcessorTestCase):
    """Integration tests for all processors."""

    def test_all_processors_implement_base_interface(self):
        """Test that all processors implement the base interface."""
        processors = [TextProcessor(), HTMLProcessor(), EmailProcessor(), PDFProcessor()]

        for processor in processors:
            # Check that processor has required methods
            self.assertTrue(hasattr(processor, 'process_document'))
            self.assertTrue(hasattr(processor, '_extract_content'))
            self.assertTrue(hasattr(processor, '_preprocess_content'))
            self.assertTrue(hasattr(processor, '_chunk_content'))
            self.assertTrue(hasattr(processor, '_upload_chunks'))

    def test_processor_configuration(self):
        """Test processor configuration handling."""
        processor = TextProcessor()

        # Test with default config
        config = processor._get_config(self.mock_context)
        self.assertIn('chunking_strategy', config)
        self.assertIn('max_chunk_size', config)
        self.assertIn('chunk_overlap', config)

    def test_error_handling(self):
        """Test error handling across processors."""
        processors = [TextProcessor(), HTMLProcessor(), EmailProcessor(), PDFProcessor()]

        for processor in processors:
            # Test with invalid file path
            with patch.object(self.mock_item, 'download', return_value="/invalid/path"):
                with self.assertRaises(Exception):
                    processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)


if __name__ == '__main__':
    # Run all tests
    test_classes = [
        TextProcessorTest,
        HTMLProcessorTest,
        EmailProcessorTest,
        PDFProcessorTest,
        ProcessorIntegrationTest,
    ]

    from pipeline.tests.test_framework import TestRunner

    result = TestRunner.run_all_tests(test_classes)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )


