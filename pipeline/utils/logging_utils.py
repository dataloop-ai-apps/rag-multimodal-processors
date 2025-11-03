"""
Basic logging and error handling utilities.
"""

import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime


class ProcessorLogger:
    """Basic logger for processors."""

    def __init__(self, processor_type: str):
        """
        Initialize logger for a processor type.

        Args:
            processor_type: Type of processor
        """
        self.processor_type = processor_type
        self.logger = logging.getLogger(f'{processor_type}-processor')

        # Set up basic logging if not already configured
        if not self.logger.handlers:
            self._setup_logging()

    def _setup_logging(self):
        """Set up basic logging configuration."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        context = self._format_context(**kwargs)
        self.logger.info(f"{message} {context}")

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        context = self._format_context(**kwargs)
        self.logger.warning(f"{message} {context}")

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        context = self._format_context(**kwargs)
        self.logger.error(f"{message} {context}")

    def _format_context(self, **kwargs) -> str:
        """Format context information."""
        if not kwargs:
            return ""

        context_parts = []
        for key, value in kwargs.items():
            context_parts.append(f"{key}={value}")

        return f"| {' '.join(context_parts)}"


class ErrorHandler:
    """Basic error handler for processors."""

    def __init__(self, processor_type: str):
        """
        Initialize error handler.

        Args:
            processor_type: Type of processor
        """
        self.processor_type = processor_type
        self.logger = ProcessorLogger(processor_type)

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Handle an error and return a user-friendly message.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        error_message = str(error)

        # Log detailed error information
        self.logger.error(f"Error occurred", error_type=error_type, error_message=error_message, **context)

        # Log stack trace for debugging
        self.logger.error(f"Stack trace: {traceback.format_exc()}")

        # Return user-friendly message
        if isinstance(error, FileNotFoundError):
            return f"File not found: {error_message}"
        elif isinstance(error, PermissionError):
            return f"Permission denied: {error_message}"
        elif isinstance(error, ValueError):
            return f"Invalid value: {error_message}"
        elif isinstance(error, MemoryError):
            return f"Out of memory: File too large to process"
        else:
            return f"Processing error: {error_message}"

    def handle_file_error(self, file_path: str, error: Exception) -> str:
        """
        Handle file-specific errors.

        Args:
            file_path: Path to the file that caused the error
            error: The exception that occurred

        Returns:
            User-friendly error message
        """
        context = {'file_path': file_path, 'processor_type': self.processor_type}
        return self.handle_error(error, context)

    def handle_processing_error(self, item_id: str, stage: str, error: Exception) -> str:
        """
        Handle processing-specific errors.

        Args:
            item_id: ID of the item being processed
            stage: Processing stage where error occurred
            error: The exception that occurred

        Returns:
            User-friendly error message
        """
        context = {'item_id': item_id, 'stage': stage, 'processor_type': self.processor_type}
        return self.handle_error(error, context)


class ValidationError(Exception):
    """Exception for validation errors."""

    pass


class FileValidator:
    """Basic file validation utilities."""

    @staticmethod
    def validate_file_type(file_path: str, expected_mime_type: str) -> bool:
        """
        Validate file type matches expected MIME type.

        Args:
            file_path: Path to the file
            expected_mime_type: Expected MIME type

        Returns:
            True if valid, False otherwise
        """
        try:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type == expected_mime_type
        except Exception:
            return False

    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: int = 100) -> bool:
        """
        Validate file size is within limits.

        Args:
            file_path: Path to the file
            max_size_mb: Maximum size in MB

        Returns:
            True if valid, False otherwise
        """
        try:
            import os

            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            return file_size <= max_size_bytes
        except Exception:
            return False

    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """
        Validate file exists.

        Args:
            file_path: Path to the file

        Returns:
            True if exists, False otherwise
        """
        try:
            import os

            return os.path.exists(file_path)
        except Exception:
            return False


