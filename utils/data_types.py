"""
Data structures for extracted multimodal content.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ImageContent:
    """Represents an extracted image."""

    path: str
    caption: Optional[str] = None
    page_number: Optional[int] = None
    bbox: Optional[tuple] = None
    format: Optional[str] = None
    size: Optional[tuple] = None
    data: Optional[bytes] = None

    def to_dict(self):
        return {
            'path': self.path,
            'caption': self.caption,
            'page_number': self.page_number,
            'bbox': self.bbox,
            'format': self.format,
            'size': self.size,
        }


@dataclass
class TableContent:
    """Represents an extracted table."""

    data: Any
    markdown: Optional[str] = None
    html: Optional[str] = None
    page_number: Optional[int] = None
    location: Optional[dict] = None

    def to_dict(self):
        return {
            'markdown': self.markdown,
            'html': self.html,
            'page_number': self.page_number,
            'location': self.location,
        }
