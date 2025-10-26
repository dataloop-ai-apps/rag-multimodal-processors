"""
Shared metadata management for chunk items across all processors.
Provides a standardized way to create and manage metadata for document chunks.
"""

import dtlpy as dl
from typing import Dict, Any


class ChunkMetadata:
    """
    Shared metadata class for all document processors.
    Creates standardized metadata structure for chunk items with minimal reference info.
    """
    
    @staticmethod
    def create_for_chunk(
        original_item: dl.Item,
        chunk_index: int,
        total_chunks: int) -> Dict[str, Any]:
        """
        Create standardized metadata for a single chunk item.
        
        Chunk items contain only reference information back to the original document.
        Processing details are stored on the original document item.
        
        Args:
            original_item (dl.Item): Original document item that was processed
            chunk_index (int): Index of this chunk (1-based)
            total_chunks (int): Total number of chunks created from the document
        
        Returns:
            Dict[str, Any]: Metadata structure for chunk item
        
        Example:
            >>> metadata = ChunkMetadata.create_for_chunk(item, 1, 50)
        """
        chunk_metadata = {
            'document': original_item.name,
            'document_type': original_item.mimetype,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'original_item_id': original_item.id,
            'original_dataset_id': original_item.dataset.id,
        }
        
        # Return in Dataloop's metadata structure format
        return {
            'user': chunk_metadata
        }

