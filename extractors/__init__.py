"""
Extractors package for extracting content from items.
Handles OCR, transcription, captioning, and other extraction methods.
"""

from .ocr_extractor import OCRExtractor

# Future extractors (uncomment as implemented):
# from .audio_extractor import AudioExtractor
# from .caption_extractor import CaptionExtractor

__all__ = ['OCRExtractor']

