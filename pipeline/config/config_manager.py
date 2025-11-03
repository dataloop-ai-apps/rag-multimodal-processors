"""
Unified configuration system for all processors.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Base configuration for all processors."""

    chunking_strategy: str = 'recursive'
    max_chunk_size: int = 300
    chunk_overlap: int = 20
    to_correct_spelling: bool = False
    enable_logging: bool = True
    log_level: str = 'INFO'


@dataclass
class PDFConfig(BaseConfig):
    """PDF-specific configuration."""

    ocr_from_images: bool = False
    ocr_integration_method: str = 'append_to_page'
    use_markdown_extraction: bool = False


@dataclass
class HTMLConfig(BaseConfig):
    """HTML-specific configuration."""

    preserve_structure: bool = True
    extract_links: bool = True


@dataclass
class EmailConfig(BaseConfig):
    """Email-specific configuration."""

    include_attachments: bool = False
    extract_headers: bool = True


@dataclass
class TextConfig(BaseConfig):
    """Text-specific configuration."""

    detect_encoding: bool = True
    preserve_csv_structure: bool = True


class ConfigManager:
    """Manages configuration for all processors."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = {}

        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file {self.config_path}: {e}")

        # Override with environment variables
        config = self._load_from_env(config)

        return config

    def _load_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_mappings = {
            'CHUNKING_STRATEGY': ('base', 'chunking_strategy'),
            'MAX_CHUNK_SIZE': ('base', 'max_chunk_size'),
            'CHUNK_OVERLAP': ('base', 'chunk_overlap'),
            'LOG_LEVEL': ('base', 'log_level'),
            'PDF_OCR_FROM_IMAGES': ('processors', 'pdf', 'ocr_from_images'),
            'PDF_USE_MARKDOWN': ('processors', 'pdf', 'use_markdown_extraction'),
            'HTML_PRESERVE_STRUCTURE': ('processors', 'html', 'preserve_structure'),
            'EMAIL_INCLUDE_ATTACHMENTS': ('processors', 'eml', 'include_attachments'),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if env_var in ['MAX_CHUNK_SIZE', 'CHUNK_OVERLAP']:
                    value = int(value)
                elif env_var in [
                    'PDF_OCR_FROM_IMAGES',
                    'PDF_USE_MARKDOWN',
                    'HTML_PRESERVE_STRUCTURE',
                    'EMAIL_INCLUDE_ATTACHMENTS',
                ]:
                    value = value.lower() in ('true', '1', 'yes', 'on')

                # Set nested config value
                self._set_nested_config(config, config_path, value)

        return config

    def _set_nested_config(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a nested configuration value."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def get_config(self, processor_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific processor type.

        Args:
            processor_type: Type of processor ('pdf', 'html', 'text', 'eml')

        Returns:
            Configuration dictionary
        """
        base_config = self._config.get('base', {})
        processor_config = self._config.get('processors', {}).get(processor_type, {})

        # Merge base and processor-specific config
        config = {**base_config, **processor_config}

        # Add processor type
        config['processor_type'] = processor_type

        return config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['chunking_strategy', 'max_chunk_size', 'chunk_overlap']

        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required config field: {field}")
                return False

        # Validate chunking strategy
        valid_strategies = ['recursive', 'fixed-size', 'nltk-sentence', 'nltk-paragraphs', '1-chunk']
        if config['chunking_strategy'] not in valid_strategies:
            print(f"Error: Invalid chunking strategy: {config['chunking_strategy']}")
            return False

        # Validate numeric fields
        if not isinstance(config['max_chunk_size'], int) or config['max_chunk_size'] <= 0:
            print(f"Error: max_chunk_size must be a positive integer")
            return False

        if not isinstance(config['chunk_overlap'], int) or config['chunk_overlap'] < 0:
            print(f"Error: chunk_overlap must be a non-negative integer")
            return False

        return True

    def save_config(self, config: Dict[str, Any], processor_type: str):
        """
        Save configuration for a processor type.

        Args:
            config: Configuration to save
            processor_type: Type of processor
        """
        if 'processors' not in self._config:
            self._config['processors'] = {}

        self._config['processors'][processor_type] = config

        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save config file {self.config_path}: {e}")


# Global configuration manager instance
config_manager = ConfigManager()
