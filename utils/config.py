"""
Simple configuration with validation.

This module provides a flat, easy-to-understand configuration class.
All settings are in one place with straightforward validation.
"""
from dataclasses import dataclass, asdict
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
    use_markdown_extraction: bool = False
    extract_images: bool = True
    extract_tables: bool = True

    # OCR settings
    ocr_from_images: bool = False
    ocr_method: Literal['local', 'batch', 'auto'] = 'local'
    ocr_model_id: Optional[str] = None
    ocr_integration_method: Literal['append_to_page', 'separate_chunks', 'combine_all'] = 'append_to_page'

    # Chunking settings
    chunking_strategy: Literal['recursive', 'fixed-size', 'nltk-sentence', 'nltk-paragraphs', '1-chunk'] = 'recursive'
    max_chunk_size: int = 300
    chunk_overlap: int = 40

    # Cleaning settings
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    to_correct_spelling: bool = False

    # LLM settings
    llm_model_id: Optional[str] = None
    generate_summary: bool = False
    extract_entities: bool = False
    translate: bool = False
    target_language: str = 'English'

    # Vision settings
    vision_model_id: Optional[str] = None

    # Upload settings
    remote_path: str = '/chunks'

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
        if self.ocr_from_images and self.ocr_method in ('batch', 'auto') and not self.ocr_model_id:
            errors.append("ocr_model_id is required when ocr_method is 'batch' or 'auto'")

        # Validate LLM settings
        if (self.generate_summary or self.extract_entities or self.translate) and not self.llm_model_id:
            errors.append("llm_model_id is required when generate_summary, extract_entities, or translate is enabled")

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
                'ocr_from_images': True,
                'to_correct_spelling': True,
                'use_markdown_extraction': False,
                'unknown_key': 'ignored'  # Will be ignored
            })
        """
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return asdict(self)
