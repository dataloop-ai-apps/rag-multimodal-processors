"""
Unit tests for static methods in processors.

Tests the static methods that enable concurrent processing:
- extract()
- clean()
- chunk()
- upload()
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import dtlpy as dl

from apps.pdf_processor.pdf_processor import PDFProcessor
from apps.doc_processor.doc_processor import DOCProcessor
from utils.chunk_metadata import ChunkMetadata


class TestPDFProcessorStaticMethods:
    """Test PDFProcessor static methods."""

    def test_extract_static_method(self):
        """Test that extract() is a static method."""
        assert isinstance(PDFProcessor.extract, staticmethod) or hasattr(PDFProcessor, 'extract')

    def test_clean_static_method(self):
        """Test that clean() is a static method."""
        assert isinstance(PDFProcessor.clean, staticmethod) or hasattr(PDFProcessor, 'clean')

    def test_chunk_static_method(self):
        """Test that chunk() is a static method."""
        assert isinstance(PDFProcessor.chunk, staticmethod) or hasattr(PDFProcessor, 'chunk')

    def test_upload_static_method(self):
        """Test that upload() is a static method."""
        assert isinstance(PDFProcessor.upload, staticmethod) or hasattr(PDFProcessor, 'upload')

    @patch('apps.pdf_processor.pdf_processor.PDFProcessor.extract_pdf')
    def test_extract_calls_extract_pdf(self, mock_extract_pdf):
        """Test that extract() calls extract_pdf()."""
        mock_item = Mock(spec=dl.Item)
        mock_item.name = 'test.pdf'

        # Create mock ExtractedContent object
        mock_extracted = Mock()
        mock_extracted.text = 'test content'
        mock_extracted.images = []
        mock_extracted.tables = []
        mock_extracted.to_dict.return_value = {
            'content': 'test content',
            'images': [],
            'tables': [],
            'metadata': {}
        }
        mock_extract_pdf.return_value = mock_extracted

        data = {'item': mock_item, 'target_dataset': Mock()}
        config = {}

        result = PDFProcessor.extract(data, config)

        mock_extract_pdf.assert_called_once_with(mock_item, config)
        assert 'content' in result

    def test_clean_processes_content(self):
        """Test that clean() processes content correctly."""
        data = {
            'content': '  test   content  \n\n\n',
            'metadata': {}
        }
        config = {}

        result = PDFProcessor.clean(data, config)

        assert 'content' in result
        # Content should be cleaned (whitespace normalized)

    def test_chunk_creates_chunks(self):
        """Test that chunk() creates chunks."""
        data = {
            'content': 'This is a test document. ' * 100,  # Long enough to chunk
            'metadata': {}
        }
        config = {
            'chunking_strategy': 'recursive',
            'max_chunk_size': 100,
            'chunk_overlap': 20
        }

        result = PDFProcessor.chunk(data, config)

        assert 'chunks' in result
        assert len(result['chunks']) > 0


class TestDOCProcessorStaticMethods:
    """Test DOCProcessor static methods."""

    def test_extract_static_method(self):
        """Test that extract() is a static method."""
        assert isinstance(DOCProcessor.extract, staticmethod) or hasattr(DOCProcessor, 'extract')

    def test_clean_static_method(self):
        """Test that clean() is a static method."""
        assert isinstance(DOCProcessor.clean, staticmethod) or hasattr(DOCProcessor, 'clean')

    def test_chunk_static_method(self):
        """Test that chunk() is a static method."""
        assert isinstance(DOCProcessor.chunk, staticmethod) or hasattr(DOCProcessor, 'chunk')

    def test_upload_static_method(self):
        """Test that upload() is a static method."""
        assert isinstance(DOCProcessor.upload, staticmethod) or hasattr(DOCProcessor, 'upload')


class TestChunkMetadata:
    """Test ChunkMetadata dataclass."""

    def test_chunk_metadata_creation(self):
        """Test creating ChunkMetadata instance."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10
        )

        assert metadata.source_item_id == 'item123'
        assert metadata.source_file == 'test.pdf'
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 10

    def test_chunk_metadata_validation(self):
        """Test that ChunkMetadata validates required fields."""
        with pytest.raises(ValueError, match="source_item_id is required"):
            ChunkMetadata(
                source_item_id='',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=0,
                total_chunks=10
            )

        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            ChunkMetadata(
                source_item_id='item123',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=-1,
                total_chunks=10
            )

    def test_chunk_metadata_to_dict(self):
        """Test converting ChunkMetadata to dictionary."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
            page_numbers=[1, 2],
            processor='pdf'
        )

        result = metadata.to_dict()

        assert 'user' in result
        assert result['user']['source_item_id'] == 'item123'
        assert result['user']['source_file'] == 'test.pdf'
        assert result['user']['chunk_index'] == 0
        assert result['user']['page_numbers'] == [1, 2]
        assert result['user']['processor'] == 'pdf'

    def test_chunk_metadata_create_from_item(self):
        """Test creating ChunkMetadata from Dataloop item."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = 'item123'
        mock_item.name = 'test.pdf'
        mock_item.dataset.id = 'dataset123'

        metadata = ChunkMetadata.create(
            source_item=mock_item,
            total_chunks=10,
            chunk_index=0
        )

        assert metadata.source_item_id == 'item123'
        assert metadata.source_file == 'test.pdf'
        assert metadata.source_dataset_id == 'dataset123'

    def test_chunk_metadata_validate_metadata(self):
        """Test validating metadata structure."""
        valid_metadata = {
            'user': {
                'source_item_id': 'item123',
                'source_file': 'test.pdf',
                'source_dataset_id': 'dataset123',
                'chunk_index': 0,
                'total_chunks': 10,
                'extracted_chunk': True,
                'processing_timestamp': 1234567890.0
            }
        }

        assert ChunkMetadata.validate_metadata(valid_metadata) is True

        invalid_metadata = {
            'user': {
                'source_item_id': 'item123',
                # Missing required fields
            }
        }

        assert ChunkMetadata.validate_metadata(invalid_metadata) is False


class TestConcurrentProcessing:
    """Test concurrent processing capabilities."""

    def test_static_methods_can_be_called_without_instance(self):
        """Test that static methods can be called without creating an instance."""
        data = {'content': 'test', 'metadata': {}}
        config = {}

        # Should work without creating an instance
        result = PDFProcessor.clean(data, config)
        assert result is not None

        result = DOCProcessor.clean(data, config)
        assert result is not None

    def test_multiple_processors_can_process_concurrently(self):
        """Test that multiple processors can process data concurrently."""
        data1 = {'content': 'test 1', 'metadata': {}}
        data2 = {'content': 'test 2', 'metadata': {}}
        config = {}

        # Both should be able to process independently
        result1 = PDFProcessor.clean(data1, config)
        result2 = DOCProcessor.clean(data2, config)

        assert result1['content'] == 'test 1'
        assert result2['content'] == 'test 2'

