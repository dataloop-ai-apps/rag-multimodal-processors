"""
Simple configuration with validation.

This module provides a flat, easy-to-understand configuration class.
All settings are in one place with straightforward validation.
"""
from dataclasses import dataclass
from typing import Optional, Literal, List


@dataclass
class Config:
    """
    Single configuration class with all processing settings.

    Flat structure for simplicity - no nested config objects.
    Use `from_dict()` to create from dictionaries.
    Use `validate()` to check configuration before processing.

    Example:
        config = Config(
            use_ocr=True,
            ocr_model_id='model-123',
            max_chunk_size=500
        )
        config.validate()
    """

    # Error handling
    error_mode: Literal['stop', 'continue'] = 'continue'
    max_errors: int = 10

    # Extraction settings
    extraction_method: Literal['markdown', 'basic'] = 'markdown'
    extract_images: bool = True
    extract_tables: bool = True

    # OCR settings
    use_ocr: bool = False
    ocr_method: Literal['local', 'batch', 'auto'] = 'local'
    ocr_model_id: Optional[str] = None

    # Chunking settings
    chunking_strategy: Literal['recursive', 'fixed', 'sentence', 'none'] = 'recursive'
    max_chunk_size: int = 300
    chunk_overlap: int = 20

    # Cleaning settings
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    use_deep_clean: bool = False

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration values are invalid.

        Example:
            config = Config(max_chunk_size=-1)
            config.validate()  # Raises ValueError
        """
        errors: List[str] = []

        # Validate chunk settings
        if self.max_chunk_size <= 0:
            errors.append(f"max_chunk_size must be positive, got {self.max_chunk_size}")

        if self.chunk_overlap < 0:
            errors.append(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.max_chunk_size:
            errors.append(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"max_chunk_size ({self.max_chunk_size})"
            )

        # Validate OCR settings
        if self.use_ocr and self.ocr_method in ('batch', 'auto') and not self.ocr_model_id:
            errors.append("ocr_model_id is required when ocr_method is 'batch' or 'auto'")

        # Validate error settings
        if self.max_errors <= 0:
            errors.append(f"max_errors must be positive, got {self.max_errors}")

        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """
        Create Config from dictionary, ignoring unknown keys.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            Config instance with values from dictionary.

        Example:
            config = Config.from_dict({
                'use_ocr': True,
                'ocr_model_id': 'model-123',
                'unknown_key': 'ignored'  # Will be ignored
            })
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """
        Convert Config to dictionary.

        Returns:
            Dictionary with all configuration values.
        """
        return {
            'error_mode': self.error_mode,
            'max_errors': self.max_errors,
            'extraction_method': self.extraction_method,
            'extract_images': self.extract_images,
            'extract_tables': self.extract_tables,
            'use_ocr': self.use_ocr,
            'ocr_method': self.ocr_method,
            'ocr_model_id': self.ocr_model_id,
            'chunking_strategy': self.chunking_strategy,
            'max_chunk_size': self.max_chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'normalize_whitespace': self.normalize_whitespace,
            'remove_empty_lines': self.remove_empty_lines,
            'use_deep_clean': self.use_deep_clean,
        }
