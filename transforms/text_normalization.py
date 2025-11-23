"""
Text normalization transforms.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import re
from utils.extracted_data import ExtractedData
from utils.text_cleaning import clean_text as deep_clean


def clean(data: ExtractedData) -> ExtractedData:
    """
    Clean and normalize text content.

    Args:
        data: ExtractedData with content_text

    Returns:
        ExtractedData with cleaned_text populated
    """
    data.current_stage = "cleaning"
    content = data.content_text

    if not content:
        data.cleaned_text = ""
        return data

    # Basic cleaning
    content = content.strip()
    content = '\n'.join(line.strip() for line in content.split('\n'))

    # Replace multiple spaces with single space
    content = re.sub(r' +', ' ', content)

    # Replace multiple newlines with double newline
    content = re.sub(r'\n{3,}', '\n\n', content)

    data.cleaned_text = content
    data.metadata['cleaning_applied'] = True

    return data


def normalize_whitespace(data: ExtractedData) -> ExtractedData:
    """
    Normalize whitespace in content.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with normalized text
    """
    content = data.get_text()

    # Replace multiple spaces with single space
    content = re.sub(r' +', ' ', content)

    # Replace multiple newlines with double newline
    content = re.sub(r'\n{3,}', '\n\n', content)

    data.cleaned_text = content
    return data


def remove_empty_lines(data: ExtractedData) -> ExtractedData:
    """
    Remove empty lines from content.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with empty lines removed
    """
    content = data.get_text()
    lines = [line for line in content.split('\n') if line.strip()]
    data.cleaned_text = '\n'.join(lines)
    return data
