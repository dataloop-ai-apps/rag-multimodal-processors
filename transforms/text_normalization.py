"""
Text normalization transforms.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import logging
import re
from functools import partial

from unstructured.cleaners.core import (
    replace_unicode_quotes,
    clean,
    clean_non_ascii_chars,
    clean_ordered_bullets,
    group_broken_paragraphs,
    remove_punctuation,
)
from unstructured.documents.elements import Text

from utils.extracted_data import ExtractedData

logger = logging.getLogger("rag-preprocessor")


class TextNormalizer:
    """Text normalization operations."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def clean_basic(text: str) -> str:
        """Basic text cleaning with whitespace normalization."""
        if not text:
            return ""

        text = text.strip()
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = TextNormalizer.normalize_whitespace(text)

        return text

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """Remove empty lines from text."""
        lines = [line for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    @staticmethod
    def deep_clean(text: str) -> str:
        """
        Apply aggressive text cleaning using unstructured.io library.

        Applies: whitespace removal, dash/bullet normalization, trailing punctuation
        removal, lowercase conversion, unicode quote replacement, non-ASCII cleaning,
        broken paragraph grouping, and ordered bullet cleaning.
        """
        if not text:
            return ""

        try:
            cleaner_partial = partial(
                clean,
                extra_whitespace=True,
                dashes=True,
                bullets=True,
                trailing_punctuation=True,
                lowercase=True
            )

            cleaners = [
                cleaner_partial,
                replace_unicode_quotes,
                clean_non_ascii_chars,
                group_broken_paragraphs,
                remove_punctuation,
            ]

            element = Text(text)
            element.apply(*cleaners)

            if element.text.split() != []:
                element.text = clean_ordered_bullets(text=element.text)

            return element.text

        except Exception as e:
            logger.warning(f"Text cleaning failed: {str(e)}, returning original text")
            return text


# Transform wrappers

def clean(data: ExtractedData) -> ExtractedData:
    """Clean and normalize text content."""
    data.current_stage = "cleaning"
    data.cleaned_text = TextNormalizer.clean_basic(data.content_text)
    data.metadata['cleaning_applied'] = True
    return data


def normalize_whitespace(data: ExtractedData) -> ExtractedData:
    """Normalize whitespace in content."""
    data.cleaned_text = TextNormalizer.normalize_whitespace(data.get_text())
    return data


def remove_empty_lines(data: ExtractedData) -> ExtractedData:
    """Remove empty lines from content."""
    data.cleaned_text = TextNormalizer.remove_empty_lines(data.get_text())
    return data


def deep_clean(data: ExtractedData) -> ExtractedData:
    """Apply aggressive text cleaning. Only runs if config.use_deep_clean is True."""
    data.current_stage = "deep_cleaning"

    if not getattr(data.config, 'use_deep_clean', False):
        return data

    content = data.get_text()
    if not content:
        return data

    data.cleaned_text = TextNormalizer.deep_clean(content)
    data.metadata['deep_cleaning_applied'] = True

    return data
