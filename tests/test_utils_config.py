"""Tests for utils/config.py - Simple configuration class."""
import pytest
from utils.config import Config


class TestConfigCreation:
    """Test Config creation and defaults."""

    def test_default_config(self):
        """Config should have sensible defaults."""
        config = Config()

        assert config.error_mode == 'continue'
        assert config.max_errors == 10
        assert config.extraction_method == 'markdown'
        assert config.chunking_strategy == 'recursive'
        assert config.max_chunk_size == 300
        assert config.chunk_overlap == 20
        assert config.use_ocr is False

    def test_custom_config(self):
        """Config should accept custom values."""
        config = Config(
            error_mode='stop',
            max_chunk_size=500,
            use_ocr=True,
            ocr_model_id='model-123'
        )

        assert config.error_mode == 'stop'
        assert config.max_chunk_size == 500
        assert config.use_ocr is True
        assert config.ocr_model_id == 'model-123'


class TestConfigFromDict:
    """Test Config.from_dict() method."""

    def test_from_dict_basic(self):
        """Should create Config from dictionary."""
        config = Config.from_dict({
            'max_chunk_size': 1000,
            'use_ocr': True,
            'ocr_model_id': 'test-model'
        })

        assert config.max_chunk_size == 1000
        assert config.use_ocr is True
        assert config.ocr_model_id == 'test-model'

    def test_from_dict_ignores_unknown_keys(self):
        """Should ignore unknown keys without error."""
        config = Config.from_dict({
            'max_chunk_size': 500,
            'unknown_key': 'should_be_ignored',
            'another_unknown': 123
        })

        assert config.max_chunk_size == 500
        # Should not raise an error

    def test_from_dict_empty(self):
        """Should return default config for empty dict."""
        config = Config.from_dict({})
        default = Config()

        assert config.max_chunk_size == default.max_chunk_size
        assert config.error_mode == default.error_mode


class TestConfigValidation:
    """Test Config.validate() method."""

    def test_valid_config(self):
        """Valid config should not raise."""
        config = Config(
            max_chunk_size=500,
            chunk_overlap=50,
            max_errors=5
        )
        config.validate()  # Should not raise

    def test_invalid_chunk_size_negative(self):
        """Should reject negative chunk size."""
        config = Config(max_chunk_size=-1)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "max_chunk_size must be positive" in str(exc_info.value)

    def test_invalid_chunk_size_zero(self):
        """Should reject zero chunk size."""
        config = Config(max_chunk_size=0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "max_chunk_size must be positive" in str(exc_info.value)

    def test_invalid_overlap_negative(self):
        """Should reject negative chunk overlap."""
        config = Config(chunk_overlap=-5)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "chunk_overlap cannot be negative" in str(exc_info.value)

    def test_invalid_overlap_too_large(self):
        """Should reject overlap >= chunk size."""
        config = Config(max_chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "chunk_overlap" in str(exc_info.value)
        assert "less than" in str(exc_info.value)

    def test_invalid_ocr_without_model(self):
        """Should reject OCR enabled without model ID."""
        config = Config(use_ocr=True, ocr_model_id=None)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "ocr_model_id is required" in str(exc_info.value)

    def test_valid_ocr_with_model(self):
        """Should accept OCR enabled with model ID."""
        config = Config(use_ocr=True, ocr_model_id='model-123')
        config.validate()  # Should not raise

    def test_invalid_max_errors(self):
        """Should reject non-positive max_errors."""
        config = Config(max_errors=0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "max_errors must be positive" in str(exc_info.value)

    def test_multiple_validation_errors(self):
        """Should report multiple validation errors."""
        config = Config(
            max_chunk_size=-1,
            chunk_overlap=-5,
            max_errors=0
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_message = str(exc_info.value)
        assert "max_chunk_size" in error_message
        assert "chunk_overlap" in error_message
        assert "max_errors" in error_message


class TestConfigToDict:
    """Test Config.to_dict() method."""

    def test_to_dict_roundtrip(self):
        """Config should survive dict roundtrip."""
        original = Config(
            error_mode='stop',
            max_chunk_size=500,
            use_ocr=True,
            ocr_model_id='test-model'
        )

        config_dict = original.to_dict()
        restored = Config.from_dict(config_dict)

        assert restored.error_mode == original.error_mode
        assert restored.max_chunk_size == original.max_chunk_size
        assert restored.use_ocr == original.use_ocr
        assert restored.ocr_model_id == original.ocr_model_id

    def test_to_dict_contains_all_fields(self):
        """to_dict should include all config fields."""
        config = Config()
        config_dict = config.to_dict()

        expected_keys = {
            'error_mode', 'max_errors', 'extraction_method',
            'extract_images', 'extract_tables', 'use_ocr', 'ocr_model_id',
            'chunking_strategy', 'max_chunk_size', 'chunk_overlap',
            'normalize_whitespace', 'remove_empty_lines'
        }

        assert set(config_dict.keys()) == expected_keys
