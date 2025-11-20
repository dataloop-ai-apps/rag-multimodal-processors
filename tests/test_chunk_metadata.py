"""
Comprehensive tests for ChunkMetadata dataclass.

Tests validation, serialization, and metadata management.
"""

import pytest
import time
from unittest.mock import Mock
import dtlpy as dl

from utils.chunk_metadata import ChunkMetadata


class TestChunkMetadataValidation:
    """Test ChunkMetadata validation."""

    def test_required_fields(self):
        """Test that all required fields are validated."""
        # Valid metadata
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
        )
        assert metadata.source_item_id == 'item123'

    def test_empty_source_item_id_raises_error(self):
        """Test that empty source_item_id raises ValueError."""
        with pytest.raises(ValueError, match="source_item_id is required"):
            ChunkMetadata(
                source_item_id='',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=0,
                total_chunks=10,
            )

    def test_empty_source_file_raises_error(self):
        """Test that empty source_file raises ValueError."""
        with pytest.raises(ValueError, match="source_file is required"):
            ChunkMetadata(
                source_item_id='item123', source_file='', source_dataset_id='dataset123', chunk_index=0, total_chunks=10
            )

    def test_negative_chunk_index_raises_error(self):
        """Test that negative chunk_index raises ValueError."""
        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            ChunkMetadata(
                source_item_id='item123',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=-1,
                total_chunks=10,
            )

    def test_zero_total_chunks_raises_error(self):
        """Test that zero total_chunks raises ValueError."""
        with pytest.raises(ValueError, match="total_chunks must be at least 1"):
            ChunkMetadata(
                source_item_id='item123',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=0,
                total_chunks=0,
            )


class TestChunkMetadataSerialization:
    """Test ChunkMetadata serialization to dict."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict() includes all fields."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=5,
            total_chunks=10,
            page_numbers=[1, 2, 3],
            image_ids=['img1', 'img2'],
            processor='pdf',
            extraction_method='pymupdf',
        )

        result = metadata.to_dict()

        assert 'user' in result
        user_meta = result['user']
        assert user_meta['source_item_id'] == 'item123'
        assert user_meta['source_file'] == 'test.pdf'
        assert user_meta['chunk_index'] == 5
        assert user_meta['total_chunks'] == 10
        assert user_meta['page_numbers'] == [1, 2, 3]
        assert user_meta['image_ids'] == ['img1', 'img2']
        assert user_meta['processor'] == 'pdf'
        assert user_meta['extraction_method'] == 'pymupdf'
        assert user_meta['extracted_chunk'] is True
        assert 'processing_timestamp' in user_meta

    def test_to_dict_omits_none_fields(self):
        """Test that to_dict() omits None optional fields."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
            # No optional fields
        )

        result = metadata.to_dict()
        user_meta = result['user']

        # Optional fields should not be present if None
        assert 'page_numbers' not in user_meta or user_meta.get('page_numbers') is None
        assert 'image_ids' not in user_meta or user_meta.get('image_ids') is None

    def test_to_dict_includes_processor_specific_metadata(self):
        """Test that processor_specific_metadata is merged into result."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
            processor_specific_metadata={'custom_field': 'custom_value', 'another_field': 42},
        )

        result = metadata.to_dict()
        user_meta = result['user']

        assert user_meta['custom_field'] == 'custom_value'
        assert user_meta['another_field'] == 42


class TestChunkMetadataCreate:
    """Test ChunkMetadata.create() class method."""

    def test_create_from_item(self):
        """Test creating metadata from Dataloop item."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = 'item123'
        mock_item.name = 'test.pdf'
        mock_item.dataset.id = 'dataset123'

        metadata = ChunkMetadata.create(source_item=mock_item, total_chunks=10, chunk_index=0)

        assert metadata.source_item_id == 'item123'
        assert metadata.source_file == 'test.pdf'
        assert metadata.source_dataset_id == 'dataset123'
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 10

    def test_create_with_optional_fields(self):
        """Test creating metadata with optional fields."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = 'item123'
        mock_item.name = 'test.pdf'
        mock_item.dataset.id = 'dataset123'

        metadata = ChunkMetadata.create(
            source_item=mock_item,
            total_chunks=10,
            chunk_index=5,
            page_numbers=[1, 2],
            image_ids=['img1'],
            processor='pdf',
            extraction_method='pymupdf',
        )

        assert metadata.page_numbers == [1, 2]
        assert metadata.image_ids == ['img1']
        assert metadata.processor == 'pdf'
        assert metadata.extraction_method == 'pymupdf'

    def test_create_defaults_chunk_index_to_zero(self):
        """Test that create() defaults chunk_index to 0 if not provided."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = 'item123'
        mock_item.name = 'test.pdf'
        mock_item.dataset.id = 'dataset123'

        metadata = ChunkMetadata.create(
            source_item=mock_item,
            total_chunks=10,
            # chunk_index not provided
        )

        assert metadata.chunk_index == 0


class TestChunkMetadataValidation:
    """Test metadata validation methods."""

    def test_validate_metadata_with_user_wrapper(self):
        """Test validating metadata with Dataloop 'user' wrapper."""
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

    def test_validate_metadata_without_user_wrapper(self):
        """Test validating metadata without 'user' wrapper."""
        valid_metadata = {
            'source_item_id': 'item123',
            'source_file': 'test.pdf',
            'source_dataset_id': 'dataset123',
            'chunk_index': 0,
            'total_chunks': 10,
            'extracted_chunk': True,
            'processing_timestamp': 1234567890.0,
        }

        assert ChunkMetadata.validate_metadata(valid_metadata) is True

    def test_validate_metadata_missing_fields(self):
        """Test that validation fails for missing required fields."""
        invalid_metadata = {
            'user': {
                'source_item_id': 'item123',
                # Missing other required fields
            }
        }

        assert ChunkMetadata.validate_metadata(invalid_metadata) is False

    def test_get_base_fields(self):
        """Test getting list of base fields."""
        base_fields = ChunkMetadata.get_base_fields()

        assert 'source_item_id' in base_fields
        assert 'source_file' in base_fields
        assert 'source_dataset_id' in base_fields
        assert 'chunk_index' in base_fields
        assert 'total_chunks' in base_fields
        assert 'extracted_chunk' in base_fields
        assert 'processing_timestamp' in base_fields
