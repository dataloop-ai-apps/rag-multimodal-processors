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


@dataclass
class ExtractedContent:
    """
    Multimodal content extracted from document.
    Can contain text, images, tables, etc.
    """

    text: str = ""  # Main text content
    images: List[ImageContent] = field(default_factory=list)
    tables: List[TableContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for pipeline processing"""
        return {
            'content': self.text,
            'images': [img.to_dict() for img in self.images],
            'tables': [tbl.to_dict() for tbl in self.tables],
            'metadata': self.metadata,
        }

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_tables(self) -> bool:
        return len(self.tables) > 0
