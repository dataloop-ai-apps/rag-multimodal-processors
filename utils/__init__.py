"""
Utils infrastructure package for shared functionality.

Public API:
- ExtractedData: Central pipeline data structure
- Config: Configuration with validation
- ErrorTracker: Error tracking utilities
- ImageContent, TableContent: Data types for content extraction
- ChunkMetadata: Metadata structure for chunks

Internal utilities (use via direct import if needed):
- dataloop_helpers: upload_chunks, cleanup_temp_items_and_folder, get_or_create_target_dataset
- dataloop_model_executor: DataloopModelExecutor
"""

# Public API - core data structures and types
from .config import Config
from .errors import ErrorTracker
from .extracted_data import ExtractedData
from .data_types import ImageContent, TableContent
from .chunk_metadata import ChunkMetadata

# Internal utilities - kept importable but not in __all__
from .dataloop_helpers import get_or_create_target_dataset, upload_chunks, cleanup_temp_items_and_folder
from .dataloop_model_executor import DataloopModelExecutor

__all__ = [
    # Core data structure
    'ExtractedData',
    # Configuration
    'Config',
    # Error handling
    'ErrorTracker',
    # Data types
    'ImageContent',
    'TableContent',
    'ChunkMetadata',
]
