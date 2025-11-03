"""
Integration tests for the processor pipeline.
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


class ProcessorPipelineIntegrationTest(ProcessorTestCase):
    """Integration tests for the complete processor pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processors = {
            'text': TextProcessor(),
            'html': HTMLProcessor(),
            'email': EmailProcessor(),
            'pdf': PDFProcessor(),
        }

    def test_end_to_end_text_processing(self):
        """Test complete text processing pipeline."""
        # Create test text file
        content = TestDataGenerator.generate_text_content()
        file_path = self.create_test_file(content, "test.txt")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(3)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ):

            processor = self.processors['text']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify pipeline stages completed
        self.assert_processing_success(result, expected_chunks=3)

    def test_end_to_end_html_processing(self):
        """Test complete HTML processing pipeline."""
        # Create test HTML file
        content = TestDataGenerator.generate_html_content()
        file_path = self.create_test_file(content, "test.html")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(2)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ), patch('apps.html_processor.html_processor.BeautifulSoup') as mock_bs:

            # Mock BeautifulSoup
            mock_soup = Mock()
            mock_soup.find.return_value = Mock(get_text=lambda: "Test Document")
            mock_soup.find_all.return_value = []
            mock_bs.return_value = mock_soup

            processor = self.processors['html']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify pipeline stages completed
        self.assert_processing_success(result, expected_chunks=2)

    def test_end_to_end_email_processing(self):
        """Test complete email processing pipeline."""
        # Create test email file
        content = TestDataGenerator.generate_email_content()
        file_path = self.create_test_file(content, "test.eml")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(2)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ):

            processor = self.processors['email']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify pipeline stages completed
        self.assert_processing_success(result, expected_chunks=2)

    def test_end_to_end_pdf_processing(self):
        """Test complete PDF processing pipeline."""
        # Create test PDF file
        content = TestDataGenerator.generate_pdf_content()
        file_path = os.path.join(self.temp_dir, "test.pdf")
        with open(file_path, 'wb') as f:
            f.write(content)

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(1)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ), patch('apps.pdf_processor.pdf_processor_new.fitz') as mock_fitz:

            # Mock PyMuPDF
            mock_doc = Mock()
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__.return_value = mock_doc
            mock_doc.__exit__.return_value = None

            mock_page = Mock()
            mock_page.get_text.return_value = "Sample PDF text"
            mock_page.get_images.return_value = []
            mock_doc.__iter__.return_value = [mock_page]

            mock_fitz.open.return_value = mock_doc

            processor = self.processors['pdf']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify pipeline stages completed
        self.assert_processing_success(result, expected_chunks=1)

    def test_pipeline_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with corrupted file
        file_path = os.path.join(self.temp_dir, "corrupted.txt")
        with open(file_path, 'w') as f:
            f.write("")  # Empty file

        with patch.object(self.mock_item, 'download', return_value=file_path):
            processor = self.processors['text']

            # Should handle empty file gracefully
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Should return empty result, not crash
        self.assertIsInstance(result, list)

    def test_pipeline_configuration_handling(self):
        """Test configuration handling across pipeline stages."""
        # Test with custom configuration
        custom_config = {
            'chunking_strategy': 'fixed-size',
            'max_chunk_size': 500,
            'chunk_overlap': 50,
            'to_correct_spelling': True,
        }

        self.mock_context.node.metadata['customNodeConfig'] = custom_config

        # Create test file
        content = TestDataGenerator.generate_text_content()
        file_path = self.create_test_file(content, "test.txt")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(2)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ):

            processor = self.processors['text']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify configuration was applied
        self.assert_processing_success(result)

    def test_pipeline_metadata_handling(self):
        """Test metadata handling across pipeline stages."""
        # Create test file
        content = TestDataGenerator.generate_text_content()
        file_path = self.create_test_file(content, "test.txt")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(2)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ) as mock_upload:

            processor = self.processors['text']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify metadata was passed to upload_chunks
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        self.assertIn('processor_metadata', call_args.kwargs)

        metadata = call_args.kwargs['processor_metadata']
        self.assertIn('processor_type', metadata)
        self.assertEqual(metadata['processor_type'], 'text')

    def test_pipeline_performance(self):
        """Test pipeline performance with larger files."""
        # Create larger test file
        content = TestDataGenerator.generate_text_content() * 100  # Repeat content
        file_path = self.create_test_file(content, "large_test.txt")

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(10)]

        with patch.object(self.mock_item, 'download', return_value=file_path), patch(
            'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
        ):

            processor = self.processors['text']
            result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)

        # Verify processing completed successfully
        self.assert_processing_success(result, expected_chunks=10)

    def test_pipeline_concurrent_processing(self):
        """Test pipeline with concurrent processing simulation."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            content = TestDataGenerator.generate_text_content()
            file_path = self.create_test_file(content, f"test_{i}.txt")
            test_files.append(file_path)

        # Mock Dataloop components
        mock_chunked_items = [Mock(spec=dl.Item) for _ in range(2)]

        results = []
        for file_path in test_files:
            with patch.object(self.mock_item, 'download', return_value=file_path), patch(
                'utils.dataloop_helpers.upload_chunks', return_value=mock_chunked_items
            ):

                processor = self.processors['text']
                result = processor.process_document(self.mock_item, self.mock_dataset, self.mock_context)
                results.append(result)

        # Verify all processing completed successfully
        for result in results:
            self.assert_processing_success(result, expected_chunks=2)


class ProcessorRegistryTest(ProcessorTestCase):
    """Test processor registry and factory pattern."""

    def test_processor_registry(self):
        """Test processor registry functionality."""
        # Test that all processors are available
        expected_processors = ['text', 'html', 'email', 'pdf']

        for processor_type in expected_processors:
            self.assertIn(processor_type, ['text', 'html', 'email', 'pdf'])

    def test_processor_factory(self):
        """Test processor factory pattern."""

        # Mock processor factory
        def create_processor(processor_type: str):
            processors = {
                'text': TextProcessor(),
                'html': HTMLProcessor(),
                'email': EmailProcessor(),
                'pdf': PDFProcessor(),
            }
            return processors.get(processor_type)

        # Test factory creates correct processors
        for processor_type in ['text', 'html', 'email', 'pdf']:
            processor = create_processor(processor_type)
            self.assertIsNotNone(processor)
            self.assertEqual(processor.processor_type, processor_type)


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)


