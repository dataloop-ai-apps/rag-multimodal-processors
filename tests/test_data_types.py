"""
Tests for data type classes (ExtractedContent, ImageContent, TableContent).
"""

import pytest
from utils.data_types import ExtractedContent, ImageContent, TableContent


class TestImageContent:
    """Test ImageContent dataclass."""

    def test_image_content_creation(self):
        """Test creating ImageContent instance."""
        image = ImageContent(path='/tmp/image.png', caption='Test image', page_number=1, format='png', size=(100, 200))

        assert image.path == '/tmp/image.png'
        assert image.caption == 'Test image'
        assert image.page_number == 1
        assert image.format == 'png'
        assert image.size == (100, 200)

    def test_image_content_to_dict(self):
        """Test converting ImageContent to dictionary."""
        image = ImageContent(
            path='/tmp/image.png',
            caption='Test image',
            page_number=1,
            format='png',
            size=(100, 200),
            bbox=(10, 20, 30, 40),
        )

        result = image.to_dict()

        assert result['path'] == '/tmp/image.png'
        assert result['caption'] == 'Test image'
        assert result['page_number'] == 1
        assert result['format'] == 'png'
        assert result['size'] == (100, 200)
        assert result['bbox'] == (10, 20, 30, 40)


class TestTableContent:
    """Test TableContent dataclass."""

    def test_table_content_creation(self):
        """Test creating TableContent instance."""
        table = TableContent(
            data=[{'col1': 'val1', 'col2': 'val2'}], markdown='| col1 | col2 |\n| val1 | val2 |', page_number=1
        )

        assert len(table.data) == 1
        assert table.markdown is not None
        assert table.page_number == 1

    def test_table_content_to_dict(self):
        """Test converting TableContent to dictionary."""
        table = TableContent(
            data=[{'col1': 'val1'}], markdown='| col1 |\n| val1 |', html='<table>...</table>', page_number=1
        )

        result = table.to_dict()

        assert 'markdown' in result
        assert 'html' in result
        assert result['page_number'] == 1


class TestExtractedContent:
    """Test ExtractedContent dataclass."""

    def test_extracted_content_creation(self):
        """Test creating ExtractedContent instance."""
        content = ExtractedContent(text='Test text', images=[], tables=[], metadata={'source': 'test'})

        assert content.text == 'Test text'
        assert len(content.images) == 0
        assert len(content.tables) == 0
        assert content.metadata['source'] == 'test'

    def test_extracted_content_with_images(self):
        """Test ExtractedContent with images."""
        images = [ImageContent(path='/tmp/img1.png', page_number=1), ImageContent(path='/tmp/img2.png', page_number=2)]

        content = ExtractedContent(text='Test', images=images)

        assert len(content.images) == 2
        assert content.has_images() is True
        assert content.has_tables() is False

    def test_extracted_content_with_tables(self):
        """Test ExtractedContent with tables."""
        tables = [TableContent(data=[{'col': 'val'}], markdown='| col |\n| val |')]

        content = ExtractedContent(text='Test', tables=tables)

        assert len(content.tables) == 1
        assert content.has_tables() is True
        assert content.has_images() is False

    def test_extracted_content_to_dict(self):
        """Test converting ExtractedContent to dictionary."""
        images = [ImageContent(path='/tmp/img.png', page_number=1)]
        tables = [TableContent(data=[], markdown='| col |')]

        content = ExtractedContent(text='Test text', images=images, tables=tables, metadata={'source': 'test'})

        result = content.to_dict()

        assert result['content'] == 'Test text'
        assert len(result['images']) == 1
        assert len(result['tables']) == 1
        assert result['metadata']['source'] == 'test'
