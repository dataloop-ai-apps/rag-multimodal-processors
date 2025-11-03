"""
Base processor class for all MIME type processors.
Implements the pipeline pattern for document processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import dtlpy as dl
from dataclasses import dataclass
from enum import Enum


class ProcessingStage(Enum):
    """Processing stages in the pipeline."""

    EXTRACTION = "extraction"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    UPLOAD = "upload"


@dataclass
class ProcessingResult:
    """Result of processing a document."""

    success: bool
    chunks: List[str]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class ProcessorError(Exception):
    """Base exception for processor errors."""

    pass


class BaseProcessor(ABC):
    """
    Base class for all MIME type processors.
    Implements the pipeline pattern for document processing.
    """

    def __init__(self, processor_type: str):
        """
        Initialize the processor.

        Args:
            processor_type: Type of processor (e.g., 'pdf', 'html', 'text', 'eml')
        """
        self.processor_type = processor_type
        self.logger = logging.getLogger(f'{processor_type}-processor')

    def process_document(self, item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """
        Main entry point for document processing.
        Implements the pipeline pattern.

        Args:
            item: Document item to process
            target_dataset: Target dataset for storing chunks
            context: Processing context with configuration

        Returns:
            List of chunk items
        """
        self.logger.info(
            f"Processing {self.processor_type} | item_id={item.id} "
            f"name={item.name} mimetype={item.mimetype} "
            f"target_dataset={target_dataset.name}"
        )

        try:
            # Get configuration
            config = self._get_config(context)

            # Stage 1: Extraction
            self.logger.info(f"Stage 1: Extraction | item_id={item.id}")
            extracted_content = self._extract_content(item, config)

            # Stage 2: Preprocessing
            self.logger.info(f"Stage 2: Preprocessing | item_id={item.id}")
            processed_content = self._preprocess_content(extracted_content, config)

            # Stage 3: Chunking
            self.logger.info(f"Stage 3: Chunking | item_id={item.id}")
            chunks = self._chunk_content(processed_content, config)

            # Stage 4: Upload
            self.logger.info(f"Stage 4: Upload | item_id={item.id}")
            chunked_items = self._upload_chunks(chunks, item, target_dataset, config)

            self.logger.info(f"Processing completed | chunks={len(chunked_items)} " f"dataset={target_dataset.name}")
            return chunked_items

        except Exception as e:
            self.logger.error(f"Processing failed | item_id={item.id} error={str(e)}")
            raise ProcessorError(f"Failed to process {self.processor_type} document: {str(e)}")

    def _get_config(self, context: dl.Context) -> Dict[str, Any]:
        """Get configuration from context."""
        node = context.node
        config = node.metadata.get('customNodeConfig', {})

        # Add processor-specific defaults
        config.setdefault('chunking_strategy', 'recursive')
        config.setdefault('max_chunk_size', 300)
        config.setdefault('chunk_overlap', 20)
        config.setdefault('to_correct_spelling', False)

        return config

    @abstractmethod
    def _extract_content(self, item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from the document.

        Args:
            item: Document item
            config: Processing configuration

        Returns:
            Dictionary containing extracted content and metadata
        """
        pass

    def _preprocess_content(self, extracted_content: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the extracted content.

        Args:
            extracted_content: Content from extraction stage
            config: Processing configuration

        Returns:
            Dictionary containing preprocessed content
        """
        # Basic preprocessing - can be overridden by subclasses
        content = extracted_content.get('content', '')

        # Apply text cleaning if requested
        if config.get('to_correct_spelling', False):
            from utils.text_cleaning import clean_text

            content = clean_text(content)
            self.logger.info("Applied text cleaning")

        return {'content': content, 'metadata': extracted_content.get('metadata', {})}

    def _chunk_content(self, processed_content: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """
        Chunk the processed content.

        Args:
            processed_content: Content from preprocessing stage
            config: Processing configuration

        Returns:
            List of text chunks
        """
        from chunkers.text_chunker import TextChunker

        content = processed_content.get('content', '')
        if not content:
            self.logger.warning("No content to chunk")
            return []

        # Create chunker
        chunker = TextChunker(
            chunk_size=config.get('max_chunk_size', 300),
            chunk_overlap=config.get('chunk_overlap', 20),
            strategy=config.get('chunking_strategy', 'recursive'),
            use_markdown_splitting=config.get('use_markdown_extraction', False),
        )

        # Create chunks
        chunks = chunker.chunk(content)
        self.logger.info(f"Created {len(chunks)} chunks")

        return chunks

    def _upload_chunks(
        self, chunks: List[str], original_item: dl.Item, target_dataset: dl.Dataset, config: Dict[str, Any]
    ) -> List[dl.Item]:
        """
        Upload chunks to the target dataset.

        Args:
            chunks: List of text chunks
            original_item: Original document item
            target_dataset: Target dataset
            config: Processing configuration

        Returns:
            List of uploaded chunk items
        """
        if not chunks:
            self.logger.warning("No chunks to upload")
            return []

        from utils.dataloop_helpers import upload_chunks

        # Get processor-specific metadata
        processor_metadata = self._get_processor_metadata(config)

        # Upload chunks
        chunked_items = upload_chunks(
            chunks=chunks,
            original_item=original_item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=processor_metadata,
        )

        return chunked_items

    def _get_processor_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processor-specific metadata.

        Args:
            config: Processing configuration

        Returns:
            Dictionary with processor-specific metadata
        """
        return {
            'processor_type': self.processor_type,
            'chunking_strategy': config.get('chunking_strategy', 'recursive'),
            'max_chunk_size': config.get('max_chunk_size', 300),
            'chunk_overlap': config.get('chunk_overlap', 20),
        }
