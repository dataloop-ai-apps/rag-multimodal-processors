"""
Data structures for extracted multimodal content.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ImageContent:
    """Represents an extracted image"""

    path: str  # File path or temp path
    caption: Optional[str] = None  # Image caption/alt text
    page_number: Optional[int] = None  # Source page (for PDFs)
    bbox: Optional[tuple] = None  # Bounding box (x, y, w, h)
    format: Optional[str] = None  # png, jpg, etc.
    size: Optional[tuple] = None  # (width, height)
    data: Optional[bytes] = None  # Raw image bytes (optional)

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
    """Represents an extracted table"""

    data: Any  # pandas DataFrame or list of dicts
    markdown: Optional[str] = None  # Table as markdown
    html: Optional[str] = None  # Table as HTML
    page_number: Optional[int] = None  # Source page
    location: Optional[Dict] = None  # Position info

    def to_dict(self):
        return {
            'markdown': self.markdown,
            'html': self.html,
            'page_number': self.page_number,
            'location': self.location,
        }
