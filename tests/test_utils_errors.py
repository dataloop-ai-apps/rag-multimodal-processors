"""Tests for utils/errors.py - Simple error tracking class."""
import pytest
from utils.errors import ErrorTracker


class TestErrorTrackerCreation:
    """Test ErrorTracker creation and defaults."""

    def test_default_tracker(self):
        """ErrorTracker should start empty with sensible defaults."""
        tracker = ErrorTracker()

        assert tracker.errors == []
        assert tracker.warnings == []
        assert tracker.max_errors == 10
        assert tracker.error_mode == 'continue'

    def test_custom_tracker(self):
        """ErrorTracker should accept custom settings."""
        tracker = ErrorTracker(error_mode='stop', max_errors=5)

        assert tracker.error_mode == 'stop'
        assert tracker.max_errors == 5


class TestAddError:
    """Test ErrorTracker.add_error() method."""

    def test_add_error_records_message(self):
        """add_error should record the error message."""
        tracker = ErrorTracker()
        tracker.add_error("Something failed")

        assert len(tracker.errors) == 1
        assert "Something failed" in tracker.errors[0]

    def test_add_error_with_stage(self):
        """add_error should include stage in message."""
        tracker = ErrorTracker()
        tracker.add_error("Extraction failed", stage="extraction")

        assert "[extraction]" in tracker.errors[0]
        assert "Extraction failed" in tracker.errors[0]

    def test_add_error_continue_mode_returns_true(self):
        """In continue mode, add_error returns True until max_errors."""
        tracker = ErrorTracker(error_mode='continue', max_errors=3)

        assert tracker.add_error("Error 1") is True
        assert tracker.add_error("Error 2") is True
        assert tracker.add_error("Error 3") is False  # Hit max_errors

    def test_add_error_stop_mode_returns_false(self):
        """In stop mode, add_error always returns False."""
        tracker = ErrorTracker(error_mode='stop')

        assert tracker.add_error("First error") is False

    def test_multiple_errors_accumulated(self):
        """Multiple errors should be accumulated."""
        tracker = ErrorTracker(max_errors=10)

        tracker.add_error("Error 1")
        tracker.add_error("Error 2")
        tracker.add_error("Error 3")

        assert len(tracker.errors) == 3


class TestAddWarning:
    """Test ErrorTracker.add_warning() method."""

    def test_add_warning_records_message(self):
        """add_warning should record the warning message."""
        tracker = ErrorTracker()
        tracker.add_warning("Something might be wrong")

        assert len(tracker.warnings) == 1
        assert "Something might be wrong" in tracker.warnings[0]

    def test_add_warning_with_stage(self):
        """add_warning should include stage in message."""
        tracker = ErrorTracker()
        tracker.add_warning("Low quality", stage="ocr")

        assert "[ocr]" in tracker.warnings[0]
        assert "Low quality" in tracker.warnings[0]

    def test_warnings_dont_count_toward_max_errors(self):
        """Warnings should not count toward max_errors limit."""
        tracker = ErrorTracker(max_errors=2)

        tracker.add_warning("Warning 1")
        tracker.add_warning("Warning 2")
        tracker.add_warning("Warning 3")

        # Should still be able to add errors
        assert tracker.add_error("Error 1") is True


class TestHelperMethods:
    """Test ErrorTracker helper methods."""

    def test_has_errors_false_when_empty(self):
        """has_errors should return False when no errors."""
        tracker = ErrorTracker()
        assert tracker.has_errors() is False

    def test_has_errors_true_when_errors(self):
        """has_errors should return True when errors exist."""
        tracker = ErrorTracker()
        tracker.add_error("An error")
        assert tracker.has_errors() is True

    def test_has_warnings_false_when_empty(self):
        """has_warnings should return False when no warnings."""
        tracker = ErrorTracker()
        assert tracker.has_warnings() is False

    def test_has_warnings_true_when_warnings(self):
        """has_warnings should return True when warnings exist."""
        tracker = ErrorTracker()
        tracker.add_warning("A warning")
        assert tracker.has_warnings() is True

    def test_get_summary(self):
        """get_summary should return formatted count."""
        tracker = ErrorTracker()
        tracker.add_error("Error 1")
        tracker.add_error("Error 2")
        tracker.add_warning("Warning 1")

        summary = tracker.get_summary()
        assert "Errors: 2" in summary
        assert "Warnings: 1" in summary

    def test_get_all_messages(self):
        """get_all_messages should return errors and warnings."""
        tracker = ErrorTracker()
        tracker.add_error("Error 1")
        tracker.add_warning("Warning 1")
        tracker.add_error("Error 2")

        messages = tracker.get_all_messages()
        assert len(messages) == 3

    def test_clear(self):
        """clear should remove all errors and warnings."""
        tracker = ErrorTracker()
        tracker.add_error("Error")
        tracker.add_warning("Warning")

        tracker.clear()

        assert tracker.errors == []
        assert tracker.warnings == []
        assert tracker.has_errors() is False
        assert tracker.has_warnings() is False


class TestErrorModes:
    """Test different error handling modes."""

    def test_continue_mode_allows_multiple_errors(self):
        """Continue mode should allow errors up to max_errors."""
        tracker = ErrorTracker(error_mode='continue', max_errors=5)

        for i in range(4):
            result = tracker.add_error(f"Error {i}")
            assert result is True, f"Should continue after error {i}"

        # Fifth error should return False
        assert tracker.add_error("Error 5") is False

    def test_stop_mode_stops_immediately(self):
        """Stop mode should return False on first error."""
        tracker = ErrorTracker(error_mode='stop')

        assert tracker.add_error("First error") is False
        # Error is still recorded
        assert len(tracker.errors) == 1

    def test_warnings_never_stop_processing(self):
        """Warnings should never cause processing to stop."""
        tracker = ErrorTracker(error_mode='stop')

        # Even in stop mode, warnings don't return False
        tracker.add_warning("Warning 1")
        tracker.add_warning("Warning 2")

        assert len(tracker.warnings) == 2
        # But errors still stop
        assert tracker.add_error("Error") is False
