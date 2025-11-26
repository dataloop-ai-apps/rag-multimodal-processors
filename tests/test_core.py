"""
Core utility tests - Config, ErrorTracker, ExtractedData, ChunkMetadata.

Tests only essential functionality:
- Config creation, validation, and serialization
- ErrorTracker error/warning handling and modes
- ExtractedData pipeline flow
- ChunkMetadata creation and serialization
"""

import pytest
from unittest.mock import Mock
import dtlpy as dl

from utils.config import Config
from utils.errors import ErrorTracker
from utils.extracted_data import ExtractedData
from utils.chunk_metadata import ChunkMetadata
from utils.data_types import ImageContent


class TestConfig:
    """Essential Config tests."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = Config()
        assert config.error_mode == 'continue'
        assert config.max_chunk_size == 300
        assert config.chunking_strategy == 'recursive'

    def test_from_dict(self):
        """Config can be created from dict."""
        config = Config.from_dict({'max_chunk_size': 500, 'use_ocr': True})
        assert config.max_chunk_size == 500
        assert config.use_ocr is True

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys are ignored."""
        config = Config.from_dict({'max_chunk_size': 500, 'unknown_key': 'ignored'})
        assert config.max_chunk_size == 500

    def test_to_dict_roundtrip(self):
        """Config survives dict roundtrip."""
        original = Config(max_chunk_size=500, use_ocr=True)
        restored = Config.from_dict(original.to_dict())
        assert restored.max_chunk_size == original.max_chunk_size
        assert restored.use_ocr == original.use_ocr

    def test_validate_rejects_invalid_chunk_size(self):
        """Validation catches invalid chunk size."""
        config = Config(max_chunk_size=-1)
        with pytest.raises(ValueError, match="max_chunk_size"):
            config.validate()

    def test_validate_rejects_overlap_greater_than_size(self):
        """Validation catches overlap >= chunk size."""
        config = Config(max_chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap"):
            config.validate()

    def test_validate_rejects_batch_ocr_without_model(self):
        """Validation catches batch OCR without model ID."""
        config = Config(use_ocr=True, ocr_method='batch', ocr_model_id=None)
        with pytest.raises(ValueError, match="ocr_model_id"):
            config.validate()

    def test_validate_rejects_llm_features_without_model(self):
        """Validation catches LLM features without model ID."""
        config = Config(generate_summary=True, llm_model_id=None)
        with pytest.raises(ValueError, match="llm_model_id"):
            config.validate()


class TestErrorTracker:
    """Essential ErrorTracker tests."""

    def test_add_error_records_message(self):
        """Errors are recorded with stage prefix."""
        tracker = ErrorTracker()
        tracker.add_error("Something failed", stage="extraction")
        assert len(tracker.errors) == 1
        assert "[extraction]" in tracker.errors[0]

    def test_continue_mode_allows_errors_up_to_max(self):
        """Continue mode allows errors until max_errors."""
        tracker = ErrorTracker(error_mode='continue', max_errors=2)
        assert tracker.add_error("Error 1") is True
        assert tracker.add_error("Error 2") is False

    def test_stop_mode_stops_on_first_error(self):
        """Stop mode returns False on first error."""
        tracker = ErrorTracker(error_mode='stop')
        assert tracker.add_error("First error") is False

    def test_warnings_dont_affect_max_errors(self):
        """Warnings don't count toward max_errors."""
        tracker = ErrorTracker(max_errors=2)
        tracker.add_warning("Warning 1")
        tracker.add_warning("Warning 2")
        # First error should still succeed (warnings don't count)
        assert tracker.add_error("Error 1") is True


class TestExtractedData:
    """Essential ExtractedData tests."""

    def test_default_creation(self):
        """ExtractedData has sensible defaults."""
        data = ExtractedData()
        assert data.content_text == ""
        assert data.chunks == []
        assert data.current_stage == "init"

    def test_config_sync_with_error_tracker(self):
        """Error tracker syncs with config settings."""
        data = ExtractedData(config=Config(error_mode='stop', max_errors=5))
        assert data.errors.error_mode == 'stop'
        assert data.errors.max_errors == 5

    def test_config_from_dict(self):
        """Config can be passed as dict."""
        data = ExtractedData(config={'max_chunk_size': 1000})
        assert isinstance(data.config, Config)
        assert data.config.max_chunk_size == 1000

    def test_get_text_prefers_cleaned(self):
        """get_text returns cleaned text when available."""
        data = ExtractedData()
        data.content_text = "raw"
        data.cleaned_text = "cleaned"
        assert data.get_text() == "cleaned"

    def test_log_error_includes_stage(self):
        """log_error includes current stage in message."""
        data = ExtractedData()
        data.current_stage = "extraction"
        data.log_error("Failed")
        assert "[extraction]" in data.errors.errors[0]

    def test_has_content_checks_text_and_images(self):
        """has_content checks text, images, and tables."""
        data = ExtractedData()
        assert data.has_content() is False

        data.content_text = "text"
        assert data.has_content() is True

    def test_item_properties(self):
        """item_name and item_id work with mock item."""
        class MockItem:
            name = "test.pdf"
            id = "item-123"

        data = ExtractedData(item=MockItem())
        assert data.item_name == "test.pdf"
        assert data.item_id == "item-123"


class TestChunkMetadata:
    """Essential ChunkMetadata tests."""

    def test_creation_with_required_fields(self):
        """ChunkMetadata requires essential fields."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
        )
        assert metadata.source_item_id == 'item123'
        assert metadata.chunk_index == 0

    def test_validation_rejects_empty_item_id(self):
        """Empty source_item_id is rejected."""
        with pytest.raises(ValueError, match="source_item_id is required"):
            ChunkMetadata(
                source_item_id='',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=0,
                total_chunks=10,
            )

    def test_validation_rejects_negative_chunk_index(self):
        """Negative chunk_index is rejected."""
        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            ChunkMetadata(
                source_item_id='item123',
                source_file='test.pdf',
                source_dataset_id='dataset123',
                chunk_index=-1,
                total_chunks=10,
            )

    def test_to_dict_structure(self):
        """to_dict returns proper structure."""
        metadata = ChunkMetadata(
            source_item_id='item123',
            source_file='test.pdf',
            source_dataset_id='dataset123',
            chunk_index=0,
            total_chunks=10,
            processor='pdf',
        )
        result = metadata.to_dict()
        assert 'user' in result
        assert result['user']['source_item_id'] == 'item123'
        assert result['user']['processor'] == 'pdf'

    def test_create_from_item(self):
        """ChunkMetadata.create() works with Dataloop item."""
        mock_item = Mock(spec=dl.Item)
        mock_item.id = 'item123'
        mock_item.name = 'test.pdf'
        mock_item.dataset.id = 'dataset123'

        metadata = ChunkMetadata.create(source_item=mock_item, total_chunks=10, chunk_index=0)
        assert metadata.source_item_id == 'item123'
        assert metadata.source_file == 'test.pdf'
