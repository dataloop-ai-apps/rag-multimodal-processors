"""
Shared metadata management for chunk items across all processors.
Provides a standardized way to create and manage metadata for document chunks.
"""

import dtlpy as dl
import time
from typing import Dict, Any


class ChunkMetadata:
    """
    Shared metadata class for all document processors.
    Creates standardized metadata structure with base fields common to all chunks
    and processor-specific fields.
    """
    
    @staticmethod
    def create(
        original_item: dl.Item,
        total_chunks: int,
        processor_specific_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized metadata for chunk items.
        
        This method creates a base metadata structure that is common across all
        document processors, and allows each processor to add its own specific
        metadata on top of the base fields.
        
        Args:
            original_item (dl.Item): Original document item that was processed
            total_chunks (int): Total number of chunks created from the document
            processor_specific_metadata (Dict[str, Any], optional): Additional metadata
                specific to the processor (e.g., OCR settings, extraction method, etc.)
        
        Returns:
            Dict[str, Any]: Complete metadata structure with base and processor-specific fields
        
        Example:
            >>> processor_metadata = {
            ...     'extraction_method': 'pymupdf4llm',
            ...     'total_pages': 10,
            ...     'chunking_strategy': 'recursive'
            ... }
            >>> metadata = ChunkMetadata.create(item, 50, processor_metadata)
        """
        # Base metadata common to all chunks from all processors
        base_metadata = {
            'document': original_item.name,
            'document_type': original_item.mimetype,
            'total_chunks': total_chunks,
            'extracted_chunk': True,
            'original_item_id': original_item.id,
            'original_dataset_id': original_item.dataset.id,
            'processing_timestamp': time.time()
        }
        
        # Merge processor-specific metadata if provided
        if processor_specific_metadata:
            base_metadata.update(processor_specific_metadata)
        
        # Return in Dataloop's metadata structure format
        return {
            'user': base_metadata
        }
    
    @staticmethod
    def get_base_fields() -> list:
        """
        Get list of base metadata field names that are common to all chunks.
        
        Returns:
            list: Field names that are always present in chunk metadata
        """
        return [
            'document',
            'document_type',
            'total_chunks',
            'extracted_chunk',
            'original_item_id',
            'original_dataset_id',
            'processing_timestamp'
        ]
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata contains all required base fields.
        
        Args:
            metadata (Dict[str, Any]): Metadata dictionary to validate
            
        Returns:
            bool: True if all base fields are present, False otherwise
        """
        if 'user' not in metadata:
            return False
        
        user_metadata = metadata['user']
        base_fields = ChunkMetadata.get_base_fields()
        
        for field in base_fields:
            if field not in user_metadata:
                return False
        
        return True

