"""
Extractors package for extracting content from items.
Handles OCR, transcription, captioning, and other extraction methods.
"""

import sys
import importlib.util
from pathlib import Path

# Import from parent-level extractors.py file
# This handles the case where extractors is both a file and a package
_parent_dir = Path(__file__).parent.parent
_extractors_file = _parent_dir / 'extractors.py'

if _extractors_file.exists():
    spec = importlib.util.spec_from_file_location("extractors_module", _extractors_file)
    extractors_module = importlib.util.module_from_spec(spec)
    sys.modules['extractors_module'] = extractors_module
    spec.loader.exec_module(extractors_module)

    # Re-export the main functions and classes
    get_extractor = extractors_module.get_extractor
    ExtractedContent = extractors_module.ExtractedContent
    BaseExtractor = extractors_module.BaseExtractor
    PDFExtractor = extractors_module.PDFExtractor
    DocsExtractor = extractors_module.DocsExtractor
    EXTRACTOR_REGISTRY = extractors_module.EXTRACTOR_REGISTRY
    get_supported_types = extractors_module.get_supported_types
    register_extractor = extractors_module.register_extractor
else:
    raise ImportError("Could not find extractors.py file")

# Import from package
from .ocr_extractor import OCRExtractor

# Future extractors (uncomment as implemented):
# from .audio_extractor import AudioExtractor
# from .caption_extractor import CaptionExtractor

__all__ = [
    'OCRExtractor',
    'BaseExtractor',
    'PDFExtractor',
    'DocsExtractor',
    'get_extractor',
    'ExtractedContent',
    'EXTRACTOR_REGISTRY',
    'get_supported_types',
    'register_extractor',
]
