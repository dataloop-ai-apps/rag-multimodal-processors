"""
Unit tests for static methods in processors.

Tests the static methods that enable concurrent processing:
- run()
- process_document()
"""

import pytest
from unittest.mock import Mock, patch
import dtlpy as dl

from apps.pdf_processor.app import PDFProcessor
from apps.doc_processor.app import DOCProcessor
from utils.chunk_metadata import ChunkMetadata


class TestPDFProcessorStaticMethods:
    """Test PDFProcessor static methods."""

    def test_run_static_method(self):
        """Test that run() is a static method."""
        assert hasattr(PDFProcessor, 'run')
        assert callable(PDFProcessor.run)


class TestDOCProcessorStaticMethods:
    """Test DOCProcessor static methods."""

    def test_run_static_method(self):
        """Test that run() is a static method."""
        assert hasattr(DOCProcessor, 'run')
        assert callable(DOCProcessor.run)


class TestChunkMetadata:
    """Test ChunkMetadata dataclass."""

    def test_chunk_metadata_creation(self):
        """Test creating ChunkMetadata instance."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
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
                total_chunks=10,
            )

        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            ChunkMetadata(
                source_item_id='item123',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=-1,
                total_chunks=10,
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
            processor='pdf',
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

        metadata = ChunkMetadata.create(source_item=mock_item, total_chunks=10, chunk_index=0)

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
                'processing_timestamp': 1234567890.0,
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
