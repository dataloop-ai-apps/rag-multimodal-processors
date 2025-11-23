"""Tests for utils/extracted_data.py - Pipeline data structure."""
import pytest
from utils.extracted_data import ExtractedData
from utils.config import Config
from utils.errors import ErrorTracker
from utils.data_types import ImageContent, TableContent


class TestExtractedDataCreation:
    """Test ExtractedData creation and defaults."""

    def test_default_creation(self):
        """ExtractedData should have sensible defaults."""
        data = ExtractedData()

        assert data.item is None
        assert data.target_dataset is None
        assert isinstance(data.config, Config)
        assert data.content_text == ""
        assert data.images == []
        assert data.tables == []
        assert data.chunks == []
        assert data.current_stage == "init"

    def test_creation_with_config(self):
        """ExtractedData should accept Config object."""
        config = Config(max_chunk_size=500, error_mode='stop')
        data = ExtractedData(config=config)

        assert data.config.max_chunk_size == 500
        assert data.config.error_mode == 'stop'
        # Error tracker should sync with config
        assert data.errors.error_mode == 'stop'

    def test_creation_with_config_dict(self):
        """ExtractedData should accept config as dict."""
        data = ExtractedData(config={'max_chunk_size': 1000, 'error_mode': 'stop'})

        assert isinstance(data.config, Config)
        assert data.config.max_chunk_size == 1000
        assert data.errors.error_mode == 'stop'


class TestExtractedDataErrorTracking:
    """Test error tracking integration."""

    def test_log_error_records_stage(self):
        """log_error should include current stage."""
        data = ExtractedData()
        data.current_stage = "extraction"
        data.log_error("Something failed")

        assert len(data.errors.errors) == 1
        assert "[extraction]" in data.errors.errors[0]

    def test_log_error_returns_continue_decision(self):
        """log_error should return whether to continue."""
        data = ExtractedData(config=Config(error_mode='stop'))

        result = data.log_error("First error")
        assert result is False  # Stop mode

        data2 = ExtractedData(config=Config(error_mode='continue', max_errors=5))
        result = data2.log_error("First error")
        assert result is True  # Continue mode

    def test_log_warning(self):
        """log_warning should record warnings."""
        data = ExtractedData()
        data.current_stage = "ocr"
        data.log_warning("Low quality image")

        assert len(data.errors.warnings) == 1
        assert "[ocr]" in data.errors.warnings[0]


class TestExtractedDataContent:
    """Test content-related methods."""

    def test_get_text_returns_raw_when_no_cleaned(self):
        """get_text should return raw text when cleaned not available."""
        data = ExtractedData()
        data.content_text = "raw text"

        assert data.get_text() == "raw text"

    def test_get_text_returns_cleaned_when_available(self):
        """get_text should prefer cleaned text."""
        data = ExtractedData()
        data.content_text = "raw text"
        data.cleaned_text = "cleaned text"

        assert data.get_text() == "cleaned text"

    def test_has_content_false_when_empty(self):
        """has_content should return False when nothing extracted."""
        data = ExtractedData()
        assert data.has_content() is False

    def test_has_content_true_with_text(self):
        """has_content should return True when text exists."""
        data = ExtractedData()
        data.content_text = "some text"
        assert data.has_content() is True

    def test_has_content_true_with_images(self):
        """has_content should return True when images exist."""
        data = ExtractedData()
        data.images = [ImageContent(path="/tmp/img.png")]
        assert data.has_content() is True

    def test_has_content_true_with_tables(self):
        """has_content should return True when tables exist."""
        data = ExtractedData()
        data.tables = [TableContent(data=[], markdown="| a | b |")]
        assert data.has_content() is True

    def test_has_images(self):
        """has_images should check images list."""
        data = ExtractedData()
        assert data.has_images() is False

        data.images = [ImageContent(path="/tmp/img.png")]
        assert data.has_images() is True

    def test_has_tables(self):
        """has_tables should check tables list."""
        data = ExtractedData()
        assert data.has_tables() is False

        data.tables = [TableContent(data=[])]
        assert data.has_tables() is True

    def test_has_chunks(self):
        """has_chunks should check chunks list."""
        data = ExtractedData()
        assert data.has_chunks() is False

        data.chunks = ["chunk 1", "chunk 2"]
        assert data.has_chunks() is True


class TestExtractedDataProperties:
    """Test item-related properties."""

    def test_item_name_without_item(self):
        """item_name should return 'unknown' when no item."""
        data = ExtractedData()
        assert data.item_name == "unknown"

    def test_item_name_with_mock_item(self):
        """item_name should return item.name when available."""
        class MockItem:
            name = "test.pdf"
            id = "item-123"

        data = ExtractedData(item=MockItem())
        assert data.item_name == "test.pdf"

    def test_item_id_without_item(self):
        """item_id should return None when no item."""
        data = ExtractedData()
        assert data.item_id is None

    def test_item_id_with_mock_item(self):
        """item_id should return item.id when available."""
        class MockItem:
            name = "test.pdf"
            id = "item-123"

        data = ExtractedData(item=MockItem())
        assert data.item_id == "item-123"


class TestExtractedDataSummary:
    """Test get_summary method."""

    def test_get_summary_structure(self):
        """get_summary should return expected structure."""
        data = ExtractedData()
        data.content_text = "Hello world"
        data.cleaned_text = "hello world"
        data.images = [ImageContent(path="/tmp/img.png")]
        data.chunks = ["chunk1", "chunk2", "chunk3"]
        data.current_stage = "chunking"

        summary = data.get_summary()

        assert summary['item'] == "unknown"
        assert summary['stage'] == "chunking"
        assert summary['text_length'] == len("Hello world")
        assert summary['cleaned_length'] == len("hello world")
        assert summary['images'] == 1
        assert summary['tables'] == 0
        assert summary['chunks'] == 3
        assert summary['uploaded'] == 0
        assert 'errors' in summary


class TestExtractedDataPipelineFlow:
    """Test typical pipeline usage patterns."""

    def test_extraction_stage(self):
        """Simulate extraction stage."""
        data = ExtractedData(config=Config(extract_images=True))
        data.current_stage = "extraction"

        # Simulate extraction
        data.content_text = "Document content here"
        data.images = [
            ImageContent(path="/tmp/img1.png", page_number=1),
            ImageContent(path="/tmp/img2.png", page_number=2),
        ]
        data.metadata = {'page_count': 5, 'author': 'Test'}

        assert data.has_content()
        assert len(data.images) == 2
        assert data.metadata['page_count'] == 5

    def test_cleaning_stage(self):
        """Simulate cleaning stage."""
        data = ExtractedData()
        data.content_text = "  Document   content   "
        data.current_stage = "cleaning"

        # Simulate cleaning
        data.cleaned_text = "Document content"

        assert data.get_text() == "Document content"

    def test_chunking_stage(self):
        """Simulate chunking stage."""
        data = ExtractedData(config=Config(max_chunk_size=100))
        data.cleaned_text = "Long document text that needs to be chunked"
        data.current_stage = "chunking"

        # Simulate chunking
        data.chunks = ["Long document", "text that needs", "to be chunked"]
        data.chunk_metadata = [
            {'chunk_index': 0},
            {'chunk_index': 1},
            {'chunk_index': 2},
        ]

        assert data.has_chunks()
        assert len(data.chunks) == 3

    def test_error_during_pipeline(self):
        """Simulate error handling during pipeline."""
        data = ExtractedData(config=Config(error_mode='continue', max_errors=3))

        # Extraction succeeds
        data.current_stage = "extraction"
        data.content_text = "Some content"

        # OCR fails but continues
        data.current_stage = "ocr"
        should_continue = data.log_error("OCR model not available")
        assert should_continue is True

        # Chunking fails but continues
        data.current_stage = "chunking"
        should_continue = data.log_error("Chunking failed")
        assert should_continue is True

        # Third error hits limit
        data.current_stage = "upload"
        should_continue = data.log_error("Upload failed")
        assert should_continue is False

        # Check error summary
        assert data.errors.has_errors()
        assert len(data.errors.errors) == 3
