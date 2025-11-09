"""
Preprocessing stages for text content.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any
import re


def clean_text(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize text content.

    Args:
        data: Must contain 'content' key with text
        config: Can contain 'correct_spelling' flag

    Returns:
        data with cleaned 'content'
    """
    content = data.get('content', '')

    if not content:
        return data

    # Basic cleaning
    content = content.strip()
    content = '\n'.join(line.strip() for line in content.split('\n'))

    # Optional spell checking
    if config.get('correct_spelling', False):
        try:
            from utils.text_cleaning import clean_text as deep_clean
            content = deep_clean(content)
        except ImportError:
            print("Warning: text_cleaning utils not found, skipping spell correction")

    data['content'] = content
    data.setdefault('metadata', {})['preprocessing_applied'] = True

    return data


def normalize_whitespace(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize whitespace in content.

    Args:
        data: Must contain 'content' key
        config: Not used

    Returns:
        data with normalized whitespace
    """
    content = data.get('content', '')

    # Replace multiple spaces with single space
    content = re.sub(r' +', ' ', content)

    # Replace multiple newlines with double newline
    content = re.sub(r'\n{3,}', '\n\n', content)

    data['content'] = content
    return data


def remove_empty_lines(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove empty lines from content.

    Args:
        data: Must contain 'content' key
        config: Not used

    Returns:
        data with empty lines removed
    """
    content = data.get('content', '')
    lines = [line for line in content.split('\n') if line.strip()]
    data['content'] = '\n'.join(lines)
    return data


def truncate_content(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Truncate content to maximum length.

    Args:
        data: Must contain 'content' key
        config: Can contain 'max_content_length'

    Returns:
        data with truncated content
    """
    max_length = config.get('max_content_length')

    if max_length and max_length > 0:
        content = data.get('content', '')
        if len(content) > max_length:
            data['content'] = content[:max_length]
            data.setdefault('metadata', {})['content_truncated'] = True
            data['metadata']['original_length'] = len(content)

    return data
