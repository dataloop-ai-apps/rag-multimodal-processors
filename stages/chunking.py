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
        strategy='recursive',
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
        strategy='nltk-sentence',
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
        strategy='nltk-paragraphs',
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
        strategy='fixed-size',
    )

    chunks = chunker.chunk(content)

    data['chunks'] = chunks
    data.setdefault('metadata', {})['chunking_method'] = 'fixed-size'
    data['metadata']['chunk_count'] = len(chunks)

    return data


def chunk_recursive_with_images(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content using recursive splitting and associate images based on page numbers.

    Tracks page numbers from text markers (e.g., "--- Page X ---") and associates
    images with chunks based on matching page numbers.

    Args:
        data: Must contain 'content' and optionally 'images'
        config: Can contain 'max_chunk_size', 'chunk_overlap', 'link_images_to_chunks'

    Returns:
        data with 'chunks' list and 'chunk_metadata' list (with page numbers and image_ids)

    TODO: Add tests for:
        - Page number extraction from text
        - Image-chunk association logic
        - Handling chunks that span multiple pages
        - PDFs without page markers
    """
    from chunkers.text_chunker import TextChunker
    import re

    content = data.get('content', '')
    images = data.get('images', [])
    link_images = config.get('link_images_to_chunks', True)
    embed_images_in_chunks = config.get('embed_images_in_chunks', False)

    if not content:
        data['chunks'] = []
        data['chunk_metadata'] = []
        return data

    # If embedding images, use specialized function
    if embed_images_in_chunks:
        return chunk_with_embedded_images(data, config)

    # Extract page numbers from content
    # Look for patterns like "--- Page X ---" or "Page X" markers
    page_markers = re.finditer(r'---\s*Page\s+(\d+)\s*---', content, re.IGNORECASE)
    page_positions = []
    for match in page_markers:
        page_num = int(match.group(1))
        page_positions.append((match.start(), page_num))

    # Chunk the content
    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='recursive',
    )

    chunks = chunker.chunk(content)

    # Determine which pages each chunk belongs to
    chunk_metadata = []
    for chunk_idx, chunk in enumerate(chunks):
        # Find the position of this chunk in the original content
        chunk_start = content.find(chunk)
        if chunk_start == -1:
            # Chunk not found (might be due to overlap), try to estimate
            # Approximate position based on chunk index
            estimated_start = sum(len(c) for c in chunks[:chunk_idx])
            chunk_start = estimated_start

        # Find which page this chunk belongs to
        chunk_pages = set()
        for pos, page_num in page_positions:
            if pos <= chunk_start:
                chunk_pages.add(page_num)
            elif pos > chunk_start + len(chunk):
                break

        # If no page markers found, try to infer from content
        if not chunk_pages and page_positions:
            # Use the last page before this chunk
            for pos, page_num in reversed(page_positions):
                if pos < chunk_start:
                    chunk_pages.add(page_num)
                    break

        # Associate images with this chunk based on page numbers
        image_indices = []
        if link_images and images:
            for img_idx, img in enumerate(images):
                img_page = img.get('page_number') if isinstance(img, dict) else getattr(img, 'page_number', None)
                if img_page and img_page in chunk_pages:
                    # Image belongs to this chunk's page
                    # Store image index (will be mapped to item ID after upload)
                    image_indices.append(img_idx)

        chunk_metadata.append(
            {
                'chunk_index': chunk_idx,
                'page_numbers': sorted(list(chunk_pages)) if chunk_pages else None,
                'image_indices': image_indices,  # Will be converted to image_ids after upload
            }
        )

    data['chunks'] = chunks
    data['chunk_metadata'] = chunk_metadata
    data.setdefault('metadata', {})['chunking_method'] = 'recursive_with_images'
    data['metadata']['chunk_count'] = len(chunks)

    return data


def chunk_with_embedded_images(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chunk content with images embedded in the text at their contextual positions.

    Inserts image references/markers into the chunk text where images appear,
    creating multimodal chunks that include both text and image context.

    Args:
        data: Must contain 'content' and optionally 'images'
        config: Can contain:
            - 'max_chunk_size': Maximum chunk size
            - 'chunk_overlap': Overlap between chunks
            - 'image_marker_format': Format for image markers ('markdown', 'reference', 'inline')
            - 'image_context_before': Characters of text before image to include
            - 'image_context_after': Characters of text after image to include

    Returns:
        data with 'chunks' list (with embedded image references) and 'chunk_metadata'

    TODO: Add tests for:
        - Image embedding at correct positions
        - Different marker formats
        - Context window around images
        - Multiple images in same chunk
    """
    from chunkers.text_chunker import TextChunker
    import re

    content = data.get('content', '')
    images = data.get('images', [])

    if not content:
        data['chunks'] = []
        data['chunk_metadata'] = []
        return data

    # Extract page numbers and positions from content
    page_markers = re.finditer(r'---\s*Page\s+(\d+)\s*---', content, re.IGNORECASE)
    page_positions = []
    for match in page_markers:
        page_num = int(match.group(1))
        page_positions.append((match.start(), page_num))

    # Build image position map: (page_num, y_position) -> image_index
    # We'll insert images after page markers, ordered by y-position (top to bottom)
    image_marker_format = config.get('image_marker_format', 'markdown')
    image_context_before = config.get('image_context_before', 200)  # chars before image
    image_context_after = config.get('image_context_after', 200)  # chars after image

    # Group images by page and sort by y-position (top to bottom)
    images_by_page = {}
    for img_idx, img in enumerate(images):
        img_page = img.get('page_number') if isinstance(img, dict) else getattr(img, 'page_number', None)
        if img_page:
            if img_page not in images_by_page:
                images_by_page[img_page] = []

            # Get y-position from bbox (lower y = higher on page)
            bbox = img.get('bbox') if isinstance(img, dict) else getattr(img, 'bbox', None)
            y_pos = bbox[1] if bbox and len(bbox) >= 2 else float('inf')

            images_by_page[img_page].append((y_pos, img_idx, img))

    # Sort images by y-position within each page
    for page_num in images_by_page:
        images_by_page[page_num].sort(key=lambda x: x[0])

    # Insert image markers into content
    # We'll insert them after page markers, in order of appearance
    content_with_images = content
    offset = 0  # Track offset from insertions

    # Process each page
    for pos, page_num in page_positions:
        if page_num not in images_by_page:
            continue

        # Find insertion point (after page marker)
        page_marker_end = content.find('\n', pos)
        if page_marker_end == -1:
            page_marker_end = pos + len(f"--- Page {page_num} ---")

        # Insert images for this page
        image_markers = []
        for y_pos, img_idx, img in images_by_page[page_num]:
            # Create image marker based on format
            if image_marker_format == 'markdown':
                img_name = f"image_page{page_num}_img{img_idx}"
                marker = f"\n\n![{img_name}](image_ref_{img_idx})\n\n"
            elif image_marker_format == 'reference':
                marker = f"\n\n[IMAGE_REF_{img_idx}]\n\n"
            else:  # inline
                marker = f"\n<image_{img_idx}>\n"

            image_markers.append((page_marker_end + offset, marker, img_idx, img))

        # Insert markers in reverse order to maintain positions
        for insert_pos, marker, img_idx, img in reversed(image_markers):
            content_with_images = content_with_images[:insert_pos] + marker + content_with_images[insert_pos:]
            offset += len(marker)

    # Now chunk the content with embedded image markers
    chunker = TextChunker(
        chunk_size=config.get('max_chunk_size', 300),
        chunk_overlap=config.get('chunk_overlap', 20),
        strategy='recursive',
    )

    chunks = chunker.chunk(content_with_images)

    # Build chunk metadata with image associations
    chunk_metadata = []
    for chunk_idx, chunk in enumerate(chunks):
        # Find which images are referenced in this chunk
        image_indices = []
        for img_idx, img in enumerate(images):
            # Check if image marker appears in chunk
            if image_marker_format == 'markdown':
                if f"image_ref_{img_idx}" in chunk or f"![image_page" in chunk:
                    image_indices.append(img_idx)
            elif image_marker_format == 'reference':
                if f"[IMAGE_REF_{img_idx}]" in chunk:
                    image_indices.append(img_idx)
            else:  # inline
                if f"<image_{img_idx}>" in chunk:
                    image_indices.append(img_idx)

        # Determine page numbers from chunk content
        chunk_pages = set()
        for pos, page_num in page_positions:
            if f"Page {page_num}" in chunk:
                chunk_pages.add(page_num)

        chunk_metadata.append(
            {
                'chunk_index': chunk_idx,
                'page_numbers': sorted(list(chunk_pages)) if chunk_pages else None,
                'image_indices': image_indices,
                'has_embedded_images': len(image_indices) > 0,
            }
        )

    data['chunks'] = chunks
    data['chunk_metadata'] = chunk_metadata
    data.setdefault('metadata', {})['chunking_method'] = 'recursive_with_embedded_images'
    data['metadata']['chunk_count'] = len(chunks)
    data['metadata']['embedded_image_count'] = sum(len(m.get('image_indices', [])) for m in chunk_metadata)

    return data
