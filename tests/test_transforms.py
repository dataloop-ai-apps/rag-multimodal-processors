"""Tests for transform functions."""

import pytest
from utils.extracted_data import ExtractedData
from utils.config import Config
from utils.data_types import ImageContent
import transforms


class TestCleanTransform:
    """Tests for text cleaning transform."""

    def test_clean_strips_whitespace(self):
        data = ExtractedData(config=Config())
        data.content_text = "  Hello World  "
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello World"

    def test_clean_normalizes_spaces(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello    World"
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello World"

    def test_clean_normalizes_newlines(self):
        data = ExtractedData(config=Config(remove_empty_lines=False))
        data.content_text = "Hello\n\n\n\nWorld"
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello\n\nWorld"

    def test_clean_sets_stage(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello"
        result = transforms.clean(data)
        assert result.current_stage == "cleaning"

    def test_clean_sets_metadata(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello"
        result = transforms.clean(data)
        assert result.metadata.get('cleaning_applied') is True

    def test_clean_empty_content(self):
        data = ExtractedData(config=Config())
        data.content_text = ""
        result = transforms.clean(data)
        assert result.cleaned_text == ""

    def test_clean_respects_normalize_whitespace_config(self):
        """Test that clean() respects normalize_whitespace config."""
        # With normalization (default)
        data = ExtractedData(config=Config(normalize_whitespace=True, remove_empty_lines=False))
        data.content_text = "Hello    World"
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello World"

        # Without normalization
        data = ExtractedData(config=Config(normalize_whitespace=False, remove_empty_lines=False))
        data.content_text = "Hello    World"
        result = transforms.clean(data)
        assert "    " in result.cleaned_text  # Multiple spaces preserved

    def test_clean_respects_remove_empty_lines_config(self):
        """Test that clean() respects remove_empty_lines config."""
        # With empty line removal (default)
        data = ExtractedData(config=Config(remove_empty_lines=True))
        data.content_text = "Hello\n\n\nWorld"
        result = transforms.clean(data)
        assert result.cleaned_text == "Hello\nWorld"

        # Without empty line removal
        data = ExtractedData(config=Config(remove_empty_lines=False))
        data.content_text = "Hello\n\n\nWorld"
        result = transforms.clean(data)
        assert "\n\n" in result.cleaned_text  # Paragraph breaks preserved


class TestTextNormalizerStaticMethods:
    """Tests for TextNormalizer static methods (for direct use)."""

    def test_normalize_whitespace_basic(self):
        result = transforms.TextNormalizer.normalize_whitespace("Hello    World")
        assert result == "Hello World"

    def test_normalize_whitespace_preserves_paragraph_breaks(self):
        result = transforms.TextNormalizer.normalize_whitespace("Hello\n\nWorld")
        assert result == "Hello\n\nWorld"

    def test_normalize_whitespace_collapses_excessive_newlines(self):
        result = transforms.TextNormalizer.normalize_whitespace("Hello\n\n\n\nWorld")
        assert result == "Hello\n\nWorld"

    def test_remove_empty_lines(self):
        result = transforms.TextNormalizer.remove_empty_lines("Hello\n\n\nWorld")
        assert result == "Hello\nWorld"

    def test_clean_basic(self):
        result = transforms.TextNormalizer.clean_basic("  Hello    World  ")
        assert result == "Hello World"


class TestChunkTransform:
    """Tests for chunking transform."""

    def test_chunk_creates_chunks(self):
        data = ExtractedData(config=Config(max_chunk_size=50, chunk_overlap=10))
        data.content_text = "This is a long text that should be split into multiple chunks for testing purposes."
        result = transforms.chunk(data)
        assert len(result.chunks) > 0

    def test_chunk_sets_stage(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello World"
        result = transforms.chunk(data)
        assert result.current_stage == "chunking"

    def test_chunk_sets_metadata(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello World"
        result = transforms.chunk(data)
        assert 'chunking_strategy' in result.metadata
        assert 'chunk_count' in result.metadata

    def test_chunk_empty_content(self):
        data = ExtractedData(config=Config())
        data.content_text = ""
        result = transforms.chunk(data)
        assert result.chunks == []

    def test_chunk_uses_cleaned_text_if_available(self):
        data = ExtractedData(config=Config())
        data.content_text = "raw content"
        data.cleaned_text = "cleaned content"
        result = transforms.chunk(data)
        # Should use cleaned_text via get_text()
        assert len(result.chunks) > 0


class TestChunkWithImagesTransform:
    """Tests for chunking with image association."""

    def test_chunk_with_images_creates_chunk_metadata(self):
        data = ExtractedData(config=Config(max_chunk_size=100))
        data.content_text = "--- Page 1 ---\nSome text on page 1.\n--- Page 2 ---\nMore text on page 2."
        data.images = [
            ImageContent(path="/tmp/img1.png", page_number=1),
            ImageContent(path="/tmp/img2.png", page_number=2),
        ]
        result = transforms.chunk_with_images(data)
        assert len(result.chunk_metadata) == len(result.chunks)

    def test_chunk_with_images_sets_stage(self):
        data = ExtractedData(config=Config())
        data.content_text = "Hello World"
        result = transforms.chunk_with_images(data)
        assert result.current_stage == "chunking"

    def test_chunk_with_images_empty_content(self):
        data = ExtractedData(config=Config())
        data.content_text = ""
        result = transforms.chunk_with_images(data)
        assert result.chunks == []
        assert result.chunk_metadata == []


class TestTextChunker:
    """Tests for TextChunker static class."""

    def test_chunker_recursive_strategy(self):
        chunks = transforms.TextChunker.chunk(
            "This is some text that needs to be chunked into smaller pieces.",
            chunk_size=50, chunk_overlap=10, strategy='recursive'
        )
        assert len(chunks) > 0

    def test_chunker_fixed_strategy(self):
        chunks = transforms.TextChunker.chunk(
            "This is some text that needs to be chunked.",
            chunk_size=20, chunk_overlap=5, strategy='fixed'
        )
        assert len(chunks) > 0

    def test_chunker_none_strategy(self):
        text = "This text should not be split."
        chunks = transforms.TextChunker.chunk(text, strategy='none')
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunker_unknown_strategy_defaults_to_recursive(self):
        chunks = transforms.TextChunker.chunk("Some text to chunk.", strategy='unknown')
        assert len(chunks) > 0


class TestLLMTransforms:
    """Tests for LLM-based transforms."""

    def test_llm_chunk_semantic_without_model(self):
        data = ExtractedData(config=Config())
        data.content_text = "Some text to chunk semantically."
        result = transforms.llm_chunk_semantic(data)
        # Should log warning and return empty chunks without model
        assert result.current_stage == "llm_chunking"

    def test_llm_summarize_without_model(self):
        data = ExtractedData(config=Config())
        data.content_text = "Some text to summarize."
        result = transforms.llm_summarize(data)
        # Should skip without model
        assert result.current_stage == "summarization"


class TestTransformSignatures:
    """Tests to verify all transforms follow the correct signature."""

    def test_clean_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.clean(data)
        assert isinstance(result, ExtractedData)

    def test_deep_clean_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.deep_clean(data)
        assert isinstance(result, ExtractedData)

    def test_chunk_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.chunk(data)
        assert isinstance(result, ExtractedData)

    def test_chunk_with_images_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.chunk_with_images(data)
        assert isinstance(result, ExtractedData)

    def test_llm_chunk_semantic_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.llm_chunk_semantic(data)
        assert isinstance(result, ExtractedData)

    def test_llm_summarize_returns_extracted_data(self):
        data = ExtractedData(config=Config())
        data.content_text = "test"
        result = transforms.llm_summarize(data)
        assert isinstance(result, ExtractedData)


class TestTransformChaining:
    """Tests for chaining transforms together."""

    def test_clean_then_chunk(self):
        data = ExtractedData(config=Config(max_chunk_size=50))
        data.content_text = "  Hello    World   with   extra   spaces  "

        data = transforms.clean(data)
        data = transforms.chunk(data)

        assert data.cleaned_text == "Hello World with extra spaces"
        assert len(data.chunks) > 0

    def test_full_pipeline_simulation(self):
        data = ExtractedData(config=Config(max_chunk_size=100))
        data.content_text = """
        This is a document with multiple paragraphs.

        It has some   extra   spaces and blank lines.


        The content should be cleaned and chunked.
        """

        # Simulate pipeline - clean() now handles everything
        data = transforms.clean(data)
        data = transforms.chunk(data)

        assert data.current_stage == "chunking"
        assert len(data.chunks) > 0
        assert data.metadata.get('cleaning_applied') is True
