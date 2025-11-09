"""
Chunking stages for splitting content into chunks.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any, List


def chunk_recursive(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content using recursive character splitting.

    Args:
        data: Must contain 'content' key
        config: Can contain 'max_chunk_size', 'chunk_overlap'

    Returns:
        data with 'chunks' list added
    """
    from chunkers.text_chunker import TextChunker

    content = data.get('content', '')

    if not content:
        data['chunks'] = []
        return data

    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='recursive'
    )

    chunks = chunker.chunk(content)

    data['chunks'] = chunks
    data.setdefault('metadata', {})['chunking_method'] = 'recursive'
    data['metadata']['chunk_count'] = len(chunks)

    return data


def chunk_by_sentence(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content by sentences using NLTK.

    Args:
        data: Must contain 'content' key
        config: Can contain 'max_chunk_size', 'chunk_overlap'

    Returns:
        data with 'chunks' list added
    """
    from chunkers.text_chunker import TextChunker

    content = data.get('content', '')

    if not content:
        data['chunks'] = []
        return data

    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='nltk-sentence'
    )

    chunks = chunker.chunk(content)

    data['chunks'] = chunks
    data.setdefault('metadata', {})['chunking_method'] = 'sentence'
    data['metadata']['chunk_count'] = len(chunks)

    return data


def chunk_by_paragraph(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content by paragraphs.

    Args:
        data: Must contain 'content' key
        config: Can contain 'max_chunk_size', 'chunk_overlap'

    Returns:
        data with 'chunks' list added
    """
    from chunkers.text_chunker import TextChunker

    content = data.get('content', '')

    if not content:
        data['chunks'] = []
        return data

    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='nltk-paragraphs'
    )

    chunks = chunker.chunk(content)

    data['chunks'] = chunks
    data.setdefault('metadata', {})['chunking_method'] = 'paragraph'
    data['metadata']['chunk_count'] = len(chunks)

    return data


def no_chunking(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return entire content as single chunk.

    Args:
        data: Must contain 'content' key
        config: Not used

    Returns:
        data with single chunk in 'chunks' list
    """
    content = data.get('content', '')
    data['chunks'] = [content] if content else []
    data.setdefault('metadata', {})['chunking_method'] = 'none'
    data['metadata']['chunk_count'] = len(data['chunks'])
    return data


def chunk_fixed_size(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content using fixed size chunks.

    Args:
        data: Must contain 'content' key
        config: Can contain 'max_chunk_size', 'chunk_overlap'

    Returns:
        data with 'chunks' list added
    """
    from chunkers.text_chunker import TextChunker

    content = data.get('content', '')

    if not content:
        data['chunks'] = []
        return data

    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='fixed-size'
    )

    chunks = chunker.chunk(content)

    data['chunks'] = chunks
    data.setdefault('metadata', {})['chunking_method'] = 'fixed-size'
    data['metadata']['chunk_count'] = len(chunks)

    return data
