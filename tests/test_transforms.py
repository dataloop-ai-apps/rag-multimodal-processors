"""
Tests for transform functions.

Tests core transform functionality:
- clean() text normalization
- chunk() text splitting
- Transform chaining
"""

import pytest
from utils.extracted_data import ExtractedData
from utils.config import Config
from utils.data_types import ImageContent
import transforms


class TestCleanTransform:
    """Tests for text cleaning transform."""

    def test_clean_normalizes_text(self):
        """clean() strips whitespace, normalizes spaces and newlines."""
        data = ExtractedData(config=Config())
        data.content_text = "  Hello    World  \n\n\n\nMore text  "
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello World\nMore text"
        assert result.current_stage == "cleaning"
        assert result.metadata.get('cleaning_applied') is True

    def test_clean_respects_config_options(self):
        """clean() respects normalize_whitespace and remove_empty_lines config."""
        # With normalization disabled
        data = ExtractedData(config=Config(normalize_whitespace=False, remove_empty_lines=False))
        data.content_text = "Hello    World\n\n\nMore"
        result = transforms.clean(data)
        assert "    " in result.cleaned_text  # Multiple spaces preserved
        assert "\n\n" in result.cleaned_text  # Empty lines preserved

    def test_clean_empty_content(self):
        """clean() handles empty content."""
        data = ExtractedData(config=Config())
        data.content_text = ""
        result = transforms.clean(data)
        assert result.cleaned_text == ""


class TestChunkTransform:
    """Tests for chunking transform."""

    def test_chunk_creates_chunks(self):
        """chunk() splits text into chunks."""
        data = ExtractedData(config=Config(max_chunk_size=50, chunk_overlap=10))
        data.content_text = "This is a long text that should be split into multiple chunks for testing purposes."
        result = transforms.chunk(data)
        assert len(result.chunks) > 1
        assert result.current_stage == "chunking"
        assert 'chunk_count' in result.metadata

    def test_chunk_uses_cleaned_text(self):
        """chunk() uses cleaned_text when available."""
        data = ExtractedData(config=Config())
        data.content_text = "raw content"
        data.cleaned_text = "cleaned content"
        result = transforms.chunk(data)
        assert len(result.chunks) > 0

    def test_chunk_empty_content(self):
        """chunk() handles empty content."""
        data = ExtractedData(config=Config())
        data.content_text = ""
        result = transforms.chunk(data)
        assert result.chunks == []


class TestChunkWithImages:
    """Tests for chunking with image association."""

    def test_chunk_with_images_creates_metadata(self):
        """chunk_with_images() associates chunks with page images."""
        data = ExtractedData(config=Config(max_chunk_size=100))
        data.content_text = "--- Page 1 ---\nText on page 1.\n--- Page 2 ---\nText on page 2."
        data.images = [
            ImageContent(path="/tmp/img1.png", page_number=1),
            ImageContent(path="/tmp/img2.png", page_number=2),
        ]
        result = transforms.chunk_with_images(data)
        assert len(result.chunk_metadata) == len(result.chunks)
        assert result.current_stage == "chunking"


class TestTextChunker:
    """Tests for TextChunker strategies."""

    def test_chunker_strategies(self):
        """TextChunker supports multiple strategies."""
        text = "This is some text that needs to be chunked into pieces."

        # Recursive (default)
        chunks = transforms.TextChunker.chunk(text, chunk_size=30, strategy='recursive')
        assert len(chunks) > 0

        # Fixed
        chunks = transforms.TextChunker.chunk(text, chunk_size=20, strategy='fixed')
        assert len(chunks) > 0

        # None (no splitting)
        chunks = transforms.TextChunker.chunk(text, strategy='none')
        assert len(chunks) == 1
        assert chunks[0] == text


class TestTransformChaining:
    """Tests for chaining transforms together."""

    def test_clean_then_chunk_pipeline(self):
        """Transforms can be chained: clean -> chunk."""
        data = ExtractedData(config=Config(max_chunk_size=50))
        data.content_text = "  Hello    World   with   extra   spaces  "

        data = transforms.clean(data)
        data = transforms.chunk(data)

        assert data.cleaned_text == "Hello World with extra spaces"
        assert len(data.chunks) > 0
        assert data.metadata.get('cleaning_applied') is True


class TestTransformSignatures:
    """Verify transforms follow correct signature."""

    def test_all_transforms_return_extracted_data(self):
        """All transforms return ExtractedData."""
        data = ExtractedData(config=Config())
        data.content_text = "test content"

        assert isinstance(transforms.clean(data), ExtractedData)
        assert isinstance(transforms.chunk(data), ExtractedData)
        assert isinstance(transforms.deep_clean(data), ExtractedData)
        assert isinstance(transforms.llm_chunk_semantic(data), ExtractedData)
        assert isinstance(transforms.llm_summarize(data), ExtractedData)
