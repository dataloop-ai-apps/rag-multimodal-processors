"""
Upload stages for storing chunks in Dataloop.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any, List
from pathlib import Path
import dtlpy as dl
import pandas as pd
from utils.dataloop_helpers import upload_chunks
from utils.chunk_metadata import ChunkMetadata


def upload_to_dataloop(data: Dict[str, Any], _config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload chunks to Dataloop dataset.

    Args:
        data: Must contain 'chunks', 'item', 'target_dataset'
        config: Not used

    Returns:
        data with 'uploaded_items' added
    """
    chunks = data.get('chunks', [])
    item = data.get('item')
    target_dataset = data.get('target_dataset')
    metadata = data.get('metadata', {})

    if not chunks:
        print("Warning: No chunks to upload")
        data['uploaded_items'] = []
        return data

    if not item or not target_dataset:
        raise ValueError("Missing 'item' or 'target_dataset' in data")

    # Check if we have per-chunk metadata for bulk upload with DataFrame
    chunk_metadata_list = data.get('chunk_metadata', [])

    if chunk_metadata_list and len(chunk_metadata_list) == len(chunks):
        # Use optimized bulk upload with per-chunk metadata
        uploaded_items = upload_chunks_bulk(
            chunks=chunks,
            chunk_metadata_list=chunk_metadata_list,
            source_item=item,
            target_dataset=target_dataset,
            processor_metadata=metadata,
        )
    else:
        # Fall back to standard upload
        # should be a loop bc each item will have slightly different metadata
        uploaded_items = upload_chunks(
            chunks=chunks,
            source_item=item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=metadata,
        )

    data['uploaded_items'] = uploaded_items
    data['metadata']['uploaded_count'] = len(uploaded_items)

    return data


def upload_chunks_bulk(
    chunks: List[str],
    chunk_metadata_list: List[Dict[str, Any]],
    source_item: dl.Item,
    target_dataset: dl.Dataset,
    processor_metadata: Dict[str, Any],
) -> List[dl.Item]:
    """
    Optimized bulk upload using pandas DataFrame.
    ONE API call instead of N calls, with per-chunk metadata support.

    Args:
        chunks: List of text chunks to upload
        chunk_metadata_list: List of metadata dicts (one per chunk)
        source_item: Source document item
        target_dataset: Target dataset for chunks
        processor_metadata: Processor-specific metadata

    Returns:
        List of uploaded chunk items
    """
    records = []
    for idx, (chunk_text, chunk_meta) in enumerate(zip(chunks, chunk_metadata_list)):
        # Preserve all chunk context metadata
        chunk_context = {
            **processor_metadata,  # Start with processor metadata
            **{
                k: v
                for k, v in chunk_meta.items()
                if k not in ['chunk_index', 'page_numbers', 'image_ids', 'image_indices']
            },  # Add chunk-specific context
        }

        # Create ChunkMetadata instance with all context preserved
        metadata = ChunkMetadata.create(
            source_item=source_item,
            total_chunks=len(chunks),
            chunk_index=idx,
            page_numbers=chunk_meta.get('page_numbers'),
            image_ids=chunk_meta.get('image_ids', []),
            processor=processor_metadata.get('processor'),
            extraction_method=processor_metadata.get('extraction_method'),
            processor_specific_metadata=chunk_context,  # Include all chunk context
        ).to_dict()  # Convert to dict for Dataloop API

        base_name = Path(source_item.name).stem
        records.append(
            {
                'filename': f"{base_name}_chunk_{idx:04d}.txt",
                'text': chunk_text,
                'metadata': metadata,
                'remote_path': f'/chunks/{source_item.dir.lstrip("/")}'.replace('\\', '/'),
            }
        )

    df = pd.DataFrame(records)

    # Single bulk upload call with per-chunk metadata
    try:
        uploaded_items = target_dataset.items.upload_dataframe(
            df=df,
            item_metadata_column='metadata',
            remote_path_column='remote_path',
            local_path_column='text',
            file_name_column='filename',
            overwrite=True,
        )

        # Handle response format
        if uploaded_items is None:
            raise dl.PlatformException(f"No chunks were uploaded! Total chunks: {len(chunks)}")
        elif isinstance(uploaded_items, dl.Item):
            uploaded_items = [uploaded_items]
        else:
            uploaded_items = [item for item in uploaded_items]

        return uploaded_items
    except AttributeError:
        # Fallback if upload_dataframe is not available
        # Use standard upload_chunks instead
        return upload_chunks(
            chunks=chunks,
            source_item=source_item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=processor_metadata,
        )


def upload_with_metadata_only(data: Dict[str, Any], _config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload only metadata without creating chunk items.
    Useful for indexing or cataloging.

    Args:
        data: Must contain 'item', 'metadata'
        config: Not used

    Returns:
        data with metadata uploaded confirmation
    """
    item = data.get('item')
    metadata = data.get('metadata', {})

    if not item:
        raise ValueError("Missing 'item' in data")

    try:
        # Update item metadata
        item.metadata['user'] = item.metadata.get('user', {})
        item.metadata['user'].update(metadata)
        item.update(system_metadata=True)

        print(f"Updated metadata for item {item.id}")
        data['metadata_uploaded'] = True

    except (AttributeError, ValueError, RuntimeError) as e:
        print(f"Warning: Failed to update metadata: {e}")
        data['metadata_uploaded'] = False

    return data


def dry_run_upload(data: Dict[str, Any], _config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate upload without actually uploading.
    Useful for testing pipelines.

    Args:
        data: Must contain 'chunks'
        config: Not used

    Returns:
        data with simulated upload info
    """
    chunks = data.get('chunks', [])

    print(f"[DRY RUN] Would upload {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:5]):  # Show first 5
        preview = chunk[:100] if len(chunk) > 100 else chunk
        print(f"  Chunk {i+1}: {preview}...")

    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")

    # Simulate upload
    data['uploaded_items'] = [f"simulated_item_{i}" for i in range(len(chunks))]
    data['metadata']['dry_run'] = True
    data['metadata']['uploaded_count'] = len(chunks)

    return data
