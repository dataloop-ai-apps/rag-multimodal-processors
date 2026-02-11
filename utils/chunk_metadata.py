"""
Shared metadata management for chunk items across all processors.
Provides a standardized way to create and manage metadata for document chunks.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import dtlpy as dl


@dataclass
class ChunkMetadata:
    """
    Standardized chunk metadata with validation.

    This dataclass provides a standardized structure for chunk metadata
    across all document processors, with validation at instantiation.
    """

    # Required fields
    source_item_id: str
    source_file: str
    source_dataset_id: str
    chunk_index: int
    total_chunks: int

    # Optional fields
    page_numbers: Optional[List[int]] = None
    image_ids: Optional[List[str]] = None
    bbox: Optional[tuple] = None
    processing_timestamp: float = field(default_factory=time.time)
    processor: Optional[str] = None
    extraction_method: Optional[str] = None
    processor_specific_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate required fields at instantiation."""
        if not self.source_item_id:
            raise ValueError("source_item_id is required")
        if not self.source_file:
            raise ValueError("source_file is required")
        if not self.source_dataset_id:
            raise ValueError("source_dataset_id is required")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.total_chunks < 1:
            raise ValueError("total_chunks must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with Dataloop metadata structure.

        Returns:
            Dict[str, Any]: Metadata in Dataloop format {'user': {...}}
        """
        base_metadata = {
            'source_item_id': self.source_item_id,
            'original_item_id': self.source_item_id,  # alias for contextual_chunks compatibility
            'source_file': self.source_file,
            'source_dataset_id': self.source_dataset_id,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'extracted_chunk': True,
            'processing_timestamp': self.processing_timestamp,
        }

        # Add optional fields if present
        if self.page_numbers:
            base_metadata['page_numbers'] = self.page_numbers
        if self.image_ids:
            base_metadata['image_ids'] = self.image_ids
        if self.bbox:
            base_metadata['bbox'] = self.bbox
        if self.processor:
            base_metadata['processor'] = self.processor
        if self.extraction_method:
            base_metadata['extraction_method'] = self.extraction_method

        # Merge processor-specific metadata if provided
        if self.processor_specific_metadata:
            base_metadata.update(self.processor_specific_metadata)

        # Return in Dataloop's metadata structure format
        return {'user': base_metadata}

    @classmethod
    def create(
        cls,
        source_item: dl.Item,
        total_chunks: int,
        chunk_index: Optional[int] = None,
        processor_specific_metadata: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        image_ids: Optional[List[str]] = None,
        processor: Optional[str] = None,
        extraction_method: Optional[str] = None,
    ) -> 'ChunkMetadata':
        """
        Create ChunkMetadata instance from Dataloop item.

        Args:
            source_item: Source document item that was processed
            total_chunks: Total number of chunks created from the document
            chunk_index: Index of this chunk (0-based)
            processor_specific_metadata: Additional metadata specific to the processor
            page_numbers: Page numbers this chunk spans
            image_ids: IDs of associated image items
            processor: Processor name (e.g., 'pdf', 'doc')
            extraction_method: Extraction method used (e.g., 'pymupdf', 'pymupdf4llm')

        Returns:
            ChunkMetadata: Instance with all fields populated
        """

        # If chunk_index not provided, use 0 as default (for bulk uploads without per-chunk metadata)
        if chunk_index is None:
            chunk_index = 0

        return cls(
            source_item_id=source_item.id,
            source_file=source_item.name,
            source_dataset_id=source_item.dataset.id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            page_numbers=page_numbers,
            image_ids=image_ids,
            processor=processor,
            extraction_method=extraction_method,
            processor_specific_metadata=processor_specific_metadata,
        )
