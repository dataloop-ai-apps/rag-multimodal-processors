"""
Central data structure for pipeline processing.

ExtractedData flows through the entire processing pipeline, carrying:
- Input: source item, target dataset, configuration
- Extraction outputs: text, images, tables
- Processing results: cleaned text, chunks
- Upload results: uploaded items
- Error tracking: errors and warnings per stage
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import dtlpy as dl
except ImportError:
    dl = None  # Allow usage without dtlpy for testing

from .data_types import ImageContent, TableContent
from .config import Config
from .errors import ErrorTracker


@dataclass
class ExtractedData:
    """
    Central data structure that flows through the processing pipeline.

    This dataclass carries all data between pipeline stages:
    extraction -> cleaning -> chunking -> upload

    Example:
        data = ExtractedData(
            item=pdf_item,
            target_dataset=chunks_dataset,
            config=Config(max_chunk_size=500)
        )

        # After extraction
        data.content_text = "extracted text..."
        data.images = [ImageContent(...)]

        # After cleaning
        data.cleaned_text = "cleaned text..."

        # After chunking
        data.chunks = ["chunk 1", "chunk 2", ...]

        # After upload
        data.uploaded_items = [item1, item2, ...]
    """

    # === INPUT ===
    item: Optional[Any] = None  # dl.Item - source document
    target_dataset: Optional[Any] = None  # dl.Dataset - where to upload chunks
    config: Config = field(default_factory=Config)

    # === EXTRACTION OUTPUTS ===
    content_text: str = ""  # Raw extracted text
    images: List[ImageContent] = field(default_factory=list)
    tables: List[TableContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Document metadata

    # === PROCESSING OUTPUTS ===
    cleaned_text: str = ""  # Text after cleaning
    chunks: List[str] = field(default_factory=list)  # Text chunks
    chunk_metadata: List[Dict[str, Any]] = field(default_factory=list)  # Per-chunk metadata

    # === UPLOAD OUTPUTS ===
    uploaded_items: List[Any] = field(default_factory=list)  # dl.Item list
    uploaded_image_ids: Dict[str, str] = field(default_factory=dict)  # path -> item_id

    # === ERROR TRACKING ===
    errors: ErrorTracker = field(default_factory=ErrorTracker)
    current_stage: str = "init"

    def __post_init__(self):
        """Initialize error tracker with config settings."""
        # Handle config passed as dict
        if isinstance(self.config, dict):
            self.config = Config.from_dict(self.config)

        # Sync error tracker with config
        self.errors.error_mode = self.config.error_mode
        self.errors.max_errors = self.config.max_errors

    def log_error(self, message: str) -> bool:
        """
        Log an error and return whether to continue processing.

        Args:
            message: Error description.

        Returns:
            True if processing should continue, False if it should stop.
        """
        return self.errors.add_error(message, self.current_stage)

    def log_warning(self, message: str) -> None:
        """Log a warning (doesn't affect processing continuation)."""
        self.errors.add_warning(message, self.current_stage)

    def get_text(self) -> str:
        """
        Get the current text content (cleaned if available, else raw).

        Returns:
            Cleaned text if available, otherwise raw extracted text.
        """
        return self.cleaned_text if self.cleaned_text else self.content_text

    def has_content(self) -> bool:
        """Check if any content has been extracted."""
        return bool(self.content_text or self.images or self.tables)

    def has_images(self) -> bool:
        """Check if images were extracted."""
        return len(self.images) > 0

    def has_tables(self) -> bool:
        """Check if tables were extracted."""
        return len(self.tables) > 0

    def has_chunks(self) -> bool:
        """Check if text has been chunked."""
        return len(self.chunks) > 0

    @property
    def item_name(self) -> str:
        """Get source item name (or 'unknown' if not available)."""
        if self.item and hasattr(self.item, 'name'):
            return self.item.name
        return "unknown"

    @property
    def item_id(self) -> Optional[str]:
        """Get source item ID (or None if not available)."""
        if self.item and hasattr(self.item, 'id'):
            return self.item.id
        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get processing summary for logging/debugging.

        Returns:
            Dictionary with processing statistics.
        """
        return {
            'item': self.item_name,
            'stage': self.current_stage,
            'text_length': len(self.content_text),
            'cleaned_length': len(self.cleaned_text),
            'images': len(self.images),
            'tables': len(self.tables),
            'chunks': len(self.chunks),
            'uploaded': len(self.uploaded_items),
            'errors': self.errors.get_summary(),
        }
