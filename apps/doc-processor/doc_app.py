"""
Simple DOC/DOCX processor app.

Uses existing extractors and stages from the repo for processing Word documents.
No complex inheritance - just straightforward function composition.
"""

import logging
from typing import Dict, Any, List
import dtlpy as dl

# Import existing utilities from repo
from extractors import DocsExtractor
import stages


class DOCApp:
    """
    Simple DOCX processing application.

    Usage:
        >>> app = DOCApp(
        ...     item=docx_item,
        ...     target_dataset=chunks_dataset,
        ...     config={'max_chunk_size': 1000}
        ... )
        >>> chunks = app.run()
    """

    def __init__(
        self,
        item: dl.Item,
        target_dataset: dl.Dataset,
        config: Dict[str, Any] = None
    ):
        """
        Initialize DOC processor.

        Args:
            item: Dataloop DOCX item to process
            target_dataset: Target dataset for output chunks
            config: Processing configuration dict
        """
        self.item = item
        self.target_dataset = target_dataset
        self.config = config or {}
        self.extractor = DocsExtractor()

        # Setup logging
        log_level = self.config.get('log_level', 'INFO')
        self.logger = logging.getLogger(f"DOCApp.{item.id[:8]}")
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from DOCX file."""
        self.logger.info(f"Extracting content from: {self.item.name}")

        extracted = self.extractor.extract(self.item, self.config)

        self.logger.info(
            f"Extracted {len(extracted.text)} chars, "
            f"{len(extracted.images)} images, "
            f"{len(extracted.tables)} tables"
        )

        # Merge extracted content into data
        data.update(extracted.to_dict())
        return data

    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize text."""
        self.logger.info("Cleaning text")
        data = stages.clean_text(data, self.config)
        data = stages.normalize_whitespace(data, self.config)
        return data

    def chunk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk content based on strategy."""
        strategy = self.config.get('chunking_strategy', 'recursive')
        self.logger.info(f"Chunking with strategy: {strategy}")

        if strategy == 'recursive':
            data = stages.chunk_recursive(data, self.config)
        elif strategy == 'semantic':
            data = stages.llm_chunk_semantic(data, self.config)
        elif strategy == 'sentence':
            data = stages.chunk_by_sentence(data, self.config)
        elif strategy == 'paragraph':
            data = stages.chunk_by_paragraph(data, self.config)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        chunk_count = len(data.get('chunks', []))
        self.logger.info(f"Created {chunk_count} chunks")
        return data

    def upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload chunks to Dataloop."""
        self.logger.info("Uploading chunks to Dataloop")
        data = stages.upload_to_dataloop(data, self.config)

        uploaded_count = len(data.get('uploaded_items', []))
        self.logger.info(f"Uploaded {uploaded_count} items")
        return data

    def run(self) -> List[dl.Item]:
        """
        Execute the full processing pipeline.

        Returns:
            List of uploaded chunk items
        """
        self.logger.info(f"Starting DOC processing: {self.item.name}")

        try:
            # Initialize data with context
            data = {
                'item': self.item,
                'target_dataset': self.target_dataset
            }

            # Execute pipeline stages sequentially
            data = self.extract(data)
            data = self.clean(data)
            data = self.chunk(data)
            data = self.upload(data)

            # Return uploaded items
            uploaded = data.get('uploaded_items', [])
            self.logger.info(f"Processing complete: {len(uploaded)} chunks created")
            return uploaded

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
