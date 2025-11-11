"""
Pytest configuration for RAG Multimodal Processors tests.

This file ensures test output is visible when running tests.
"""
import sys
from pathlib import Path

# Add repository root to Python path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
