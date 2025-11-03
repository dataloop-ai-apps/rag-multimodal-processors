"""
Basic test framework for processors.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import dtlpy as dl


class ProcessorTestCase(unittest.TestCase):
    """Base test case for processor testing."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_item = self._create_mock_item()
        self.mock_dataset = self._create_mock_dataset()
        self.mock_context = self._create_mock_context()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_item(self) -> Mock:
        """Create a mock Dataloop item."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = "test_item_123"
        mock_item.name = "test_file.pdf"
        mock_item.mimetype = "application/pdf"
        mock_item.dataset = Mock()
        return mock_item

    def _create_mock_dataset(self) -> Mock:
        """Create a mock Dataloop dataset."""
        mock_dataset = Mock(spec=dl.Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset.items = Mock()
        mock_dataset.items.upload = Mock(return_value=[])
        return mock_dataset

    def _create_mock_context(self) -> Mock:
        """Create a mock Dataloop context."""
        mock_context = Mock(spec=dl.Context)
        mock_node = Mock()
        mock_node.metadata = {
            'customNodeConfig': {
                'chunking_strategy': 'recursive',
                'max_chunk_size': 300,
                'chunk_overlap': 20,
                'to_correct_spelling': False,
            }
        }
        mock_context.node = mock_node
        return mock_context

    def create_test_file(self, content: str, filename: str) -> str:
        """
        Create a test file with given content.

        Args:
            content: File content
            filename: Name of the file

        Returns:
            Path to the created file
        """
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    def assert_processing_success(self, result: List[dl.Item], expected_chunks: int = None):
        """
        Assert that processing was successful.

        Args:
            result: Processing result
            expected_chunks: Expected number of chunks
        """
        self.assertIsInstance(result, list)
        if expected_chunks is not None:
            self.assertEqual(len(result), expected_chunks)

    def assert_processing_error(self, exception: Exception, expected_error_type: type = None):
        """
        Assert that processing failed with expected error.

        Args:
            exception: The exception that was raised
            expected_error_type: Expected exception type
        """
        if expected_error_type:
            self.assertIsInstance(exception, expected_error_type)


class MockProcessor:
    """Mock processor for testing."""

    def __init__(self, processor_type: str):
        self.processor_type = processor_type
        self.logger = Mock()

    def process_document(self, item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """Mock process_document method."""
        return []


class TestDataGenerator:
    """Generate test data for different file types."""

    @staticmethod
    def generate_text_content() -> str:
        """Generate sample text content."""
        return """
        This is a sample text document for testing.
        It contains multiple paragraphs and sentences.
        
        The purpose is to test text processing functionality.
        This includes chunking and text cleaning features.
        """

    @staticmethod
    def generate_html_content() -> str:
        """Generate sample HTML content."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Document</title>
            <meta name="description" content="Test HTML document">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a paragraph with <strong>bold text</strong>.</p>
            <h2>Subheading</h2>
            <p>Another paragraph with <a href="https://example.com">a link</a>.</p>
        </body>
        </html>
        """

    @staticmethod
    def generate_csv_content() -> str:
        """Generate sample CSV content."""
        return """Name,Age,City
John Doe,30,New York
Jane Smith,25,Los Angeles
Bob Johnson,35,Chicago"""

    @staticmethod
    def generate_email_content() -> str:
        """Generate sample email content."""
        return """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 01 Jan 2024 12:00:00 +0000

This is a test email message.
It contains multiple lines of text.

Best regards,
Sender"""

    @staticmethod
    def generate_pdf_content() -> bytes:
        """Generate sample PDF content (minimal PDF)."""
        # This is a minimal PDF structure - in real tests, use proper PDF generation
        return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""


class TestRunner:
    """Test runner for processor tests."""

    @staticmethod
    def run_tests(test_class: type) -> unittest.TestResult:
        """
        Run tests for a test class.

        Args:
            test_class: Test class to run

        Returns:
            Test result
        """
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result

    @staticmethod
    def run_all_tests(test_classes: List[type]) -> unittest.TestResult:
        """
        Run all test classes.

        Args:
            test_classes: List of test classes to run

        Returns:
            Combined test result
        """
        suite = unittest.TestSuite()
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result
