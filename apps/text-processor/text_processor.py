"""
Text processor for handling text files (.txt, .md, .csv).
"""

import os
import csv
import chardet
from typing import Dict, Any, List
import dtlpy as dl
from pipeline.base.processor import BaseProcessor, ProcessorError
from pipeline.utils.logging_utils import ProcessorLogger, ErrorHandler, FileValidator


class TextProcessor(BaseProcessor):
    """
    Processor for text files (.txt, .md, .csv).
    """

    def __init__(self):
        """Initialize text processor."""
        super().__init__('text')
        self.logger = ProcessorLogger('text')
        self.error_handler = ErrorHandler('text')
        self.validator = FileValidator()

    def _extract_content(self, item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from text file.

        Args:
            item: Text file item
            config: Processing configuration

        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Download file to temporary location
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)

                # Validate file
                if not self.validator.validate_file_exists(file_path):
                    raise ProcessorError(f"File not found: {file_path}")

                if not self.validator.validate_file_size(file_path, max_size_mb=100):
                    raise ProcessorError(f"File too large: {file_path}")

                # Detect file type and extract content
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension == '.csv':
                    content, metadata = self._extract_csv_content(file_path, config)
                else:
                    content, metadata = self._extract_text_content(file_path, config)

                self.logger.info(
                    f"Extracted content", file_path=file_path, content_length=len(content), file_type=file_extension
                )

                return {'content': content, 'metadata': metadata}

        except Exception as e:
            error_msg = self.error_handler.handle_file_error(item.name, e)
            raise ProcessorError(error_msg)

    def _extract_text_content(self, file_path: str, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Extract content from text file (.txt, .md).

        Args:
            file_path: Path to the text file
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        # Detect encoding
        encoding = 'utf-8'
        if config.get('detect_encoding', True):
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected.get('encoding', 'utf-8')
                    confidence = detected.get('confidence', 0)

                    self.logger.info(f"Detected encoding", encoding=encoding, confidence=confidence)
            except Exception as e:
                self.logger.warning(f"Failed to detect encoding: {e}")

        # Read file content
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            self.logger.warning("Encoding detection failed, using utf-8 with error handling")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        # Get file metadata
        file_stats = os.stat(file_path)
        metadata = {
            'file_type': 'text',
            'encoding': encoding,
            'file_size': file_stats.st_size,
            'line_count': len(content.splitlines()),
            'word_count': len(content.split()),
        }

        return content, metadata

    def _extract_csv_content(self, file_path: str, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Extract content from CSV file.

        Args:
            file_path: Path to the CSV file
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        try:
            # Detect encoding
            encoding = 'utf-8'
            if config.get('detect_encoding', True):
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        detected = chardet.detect(raw_data)
                        encoding = detected.get('encoding', 'utf-8')
                except Exception as e:
                    self.logger.warning(f"Failed to detect CSV encoding: {e}")

            # Read CSV content
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)

            if not rows:
                return "", {'file_type': 'csv', 'rows': 0, 'columns': 0}

            # Convert CSV to text format
            if config.get('preserve_csv_structure', True):
                # Preserve CSV structure
                content = self._csv_to_structured_text(rows)
            else:
                # Flatten CSV to plain text
                content = self._csv_to_plain_text(rows)

            # Get CSV metadata
            metadata = {
                'file_type': 'csv',
                'encoding': encoding,
                'rows': len(rows),
                'columns': len(rows[0]) if rows else 0,
                'headers': rows[0] if rows else [],
            }

            return content, metadata

        except Exception as e:
            self.logger.error(f"Failed to process CSV file: {e}")
            raise ProcessorError(f"CSV processing failed: {str(e)}")

    def _csv_to_structured_text(self, rows: List[List[str]]) -> str:
        """
        Convert CSV rows to structured text format.

        Args:
            rows: List of CSV rows

        Returns:
            Structured text representation
        """
        if not rows:
            return ""

        # Use first row as headers
        headers = rows[0]
        data_rows = rows[1:]

        # Create structured text
        lines = []
        lines.append("CSV Data:")
        lines.append("=" * 50)

        for i, row in enumerate(data_rows):
            lines.append(f"Row {i + 1}:")
            for j, value in enumerate(row):
                if j < len(headers):
                    lines.append(f"  {headers[j]}: {value}")
            lines.append("")

        return "\n".join(lines)

    def _csv_to_plain_text(self, rows: List[List[str]]) -> str:
        """
        Convert CSV rows to plain text format.

        Args:
            rows: List of CSV rows

        Returns:
            Plain text representation
        """
        if not rows:
            return ""

        # Join all cells with spaces
        text_lines = []
        for row in rows:
            text_lines.append(" ".join(row))

        return "\n".join(text_lines)

    def _get_processor_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processor-specific metadata.

        Args:
            config: Processing configuration

        Returns:
            Dictionary with processor-specific metadata
        """
        metadata = super()._get_processor_metadata(config)
        metadata.update(
            {
                'processor_type': 'text',
                'detect_encoding': config.get('detect_encoding', True),
                'preserve_csv_structure': config.get('preserve_csv_structure', True),
            }
        )
        return metadata


