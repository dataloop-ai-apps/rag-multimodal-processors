"""
Text normalization transforms.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import logging
import re
from functools import partial

from unstructured.cleaners.core import (
    replace_unicode_quotes,
    clean as unstructured_clean,
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
        """Normalize whitespace: collapse multiple spaces, limit consecutive newlines."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """Remove empty lines from text (collapses paragraph breaks)."""
        lines = [line for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    @staticmethod
    def deep_clean(text: str) -> str:
        """
        Aggressive text cleaning using unstructured.io library.

        Applies: whitespace removal, dash/bullet normalization, trailing punctuation
        removal, lowercase conversion, unicode quote replacement, non-ASCII cleaning,
        broken paragraph grouping, and ordered bullet cleaning.
        """
        if not text:
            return ""

        try:
            cleaner_partial = partial(
                unstructured_clean,
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
            logger.warning(f"Deep cleaning failed: {str(e)}, returning original text")
            return text


# Transform wrappers

def clean(data: ExtractedData) -> ExtractedData:
    """
    Clean and normalize text content based on config options.

    Respects:
    - config.normalize_whitespace: Collapse multiple spaces/newlines (default: True)
    - config.remove_empty_lines: Remove blank lines (default: True)
    """
    data.current_stage = "cleaning"

    text = data.content_text
    if not text:
        data.cleaned_text = ""
        return data

    # Always strip lines
    text = text.strip()
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Optional: normalize whitespace
    if data.config.normalize_whitespace:
        text = TextNormalizer.normalize_whitespace(text)

    # Optional: remove empty lines
    if data.config.remove_empty_lines:
        text = TextNormalizer.remove_empty_lines(text)

    data.cleaned_text = text
    data.metadata['cleaning_applied'] = True
    data.metadata['normalize_whitespace'] = data.config.normalize_whitespace
    data.metadata['remove_empty_lines'] = data.config.remove_empty_lines

    return data


def deep_clean(data: ExtractedData) -> ExtractedData:
    """Apply aggressive text cleaning using unstructured.io library."""
    data.current_stage = "deep_cleaning"

    content = data.get_text()
    if not content:
        return data

    data.cleaned_text = TextNormalizer.deep_clean(content)
    data.metadata['deep_cleaning_applied'] = True

    return data
