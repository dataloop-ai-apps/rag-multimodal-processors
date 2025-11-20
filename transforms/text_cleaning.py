"""
Text cleaning utilities using unstructured.io library.
Handles text normalization, cleaning, and standardization.
"""

import logging
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

logger = logging.getLogger("rag-preprocessor")


def clean_text(text: str) -> str:
    """
    Clean a single chunk of text using unstructured.io cleaning functions.

    Applies the following cleaning steps:
    - Removes extra whitespace
    - Normalizes dashes and bullets
    - Removes trailing punctuation
    - Converts to lowercase
    - Replaces unicode quotes
    - Cleans non-ASCII characters
    - Groups broken paragraphs
    - Removes unnecessary punctuation
    - Cleans ordered bullets

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text
    """
    try:
        # Create a partial function for cleaner with specified parameters
        cleaner1_partial = partial(
            clean, extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=True, lowercase=True
        )

        cleaners = [
            cleaner1_partial,
            replace_unicode_quotes,
            clean_non_ascii_chars,
            group_broken_paragraphs,
            remove_punctuation,
        ]

        # Create a Text element and apply cleaners
        element = Text(text)
        element.apply(*cleaners)

        if element.text.split() != []:
            element.text = clean_ordered_bullets(text=element.text)

        return element.text

    except Exception as e:
        logger.warning(f"Text cleaning failed: {str(e)}, returning original text")
        return text
