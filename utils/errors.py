"""
Simple error tracking without over-abstraction.

This module provides a straightforward way to track errors and warnings
during document processing. No complex patterns - just a list of errors
and simple decision logic.
"""
from dataclasses import dataclass, field
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorTracker:
    """
    Simple error and warning tracker for pipeline processing.

    Tracks errors and warnings as simple strings, and provides
    basic decision logic for whether to continue processing.

    Example:
        tracker = ErrorTracker(error_mode='continue', max_errors=5)

        # Log an error and check if we should continue
        if not tracker.add_error("Extraction failed", "extraction"):
            return []  # Stop processing

        # Log a warning (never stops processing)
        tracker.add_warning("OCR quality low")

        # Get summary for logging
        print(tracker.get_summary())  # "Errors: 1, Warnings: 1"
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_errors: int = 10
    error_mode: str = 'continue'

    def add_error(self, message: str, stage: str = "") -> bool:
        """
        Add an error and return whether to continue processing.

        Args:
            message: Error description.
            stage: Optional processing stage (e.g., "extraction", "chunking").

        Returns:
            True if processing should continue, False if it should stop.

        Example:
            if not tracker.add_error("Failed to parse", "extraction"):
                return []  # Stop processing
        """
        error_msg = f"[{stage}] {message}" if stage else message
        self.errors.append(error_msg)
        logger.error(error_msg)

        # Decision logic: stop or continue?
        if self.error_mode == 'stop':
            return False  # Stop on first error
        else:
            # Continue if we haven't hit max_errors yet
            return len(self.errors) < self.max_errors

    def add_warning(self, message: str, stage: str = "") -> None:
        """
        Add a warning (doesn't affect processing continuation).

        Args:
            message: Warning description.
            stage: Optional processing stage.
        """
        warning_msg = f"[{stage}] {message}" if stage else message
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings have been recorded."""
        return len(self.warnings) > 0

    def get_summary(self) -> str:
        """
        Get a simple summary string for logging.

        Returns:
            Summary like "Errors: 2, Warnings: 3"
        """
        return f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"

    def get_all_messages(self) -> List[str]:
        """
        Get all errors and warnings as a single list.

        Returns:
            List of all error and warning messages.
        """
        return self.errors + self.warnings

    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
