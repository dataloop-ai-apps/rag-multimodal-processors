"""
Utils infrastructure package for shared functionality.

Public API:
- ExtractedData: Central pipeline data structure
- Config: Configuration with validation
- ErrorTracker: Error tracking utilities
- ImageContent, TableContent: Data types for content extraction
- ChunkMetadata: Metadata structure for chunks
"""

from .config import Config
from .errors import ErrorTracker
from .extracted_data import ExtractedData
from .data_types import ImageContent, TableContent
from .chunk_metadata import ChunkMetadata

__all__ = [
    'ExtractedData',
    'Config',
    'ErrorTracker',
    'ImageContent',
    'TableContent',
    'ChunkMetadata',
]
