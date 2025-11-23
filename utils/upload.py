"""
Upload transforms for storing chunks in Dataloop.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

from typing import Dict, Any, List
from pathlib import Path

import dtlpy as dl
import pandas as pd

from .dataloop_helpers import upload_chunks
from .chunk_metadata import ChunkMetadata


def upload_to_dataloop(data) -> Any:
    """
    Upload chunks to Dataloop dataset with optional image associations.

    Args:
        data: ExtractedData with chunks, item, and target_dataset

    Returns:
        ExtractedData with uploaded_items populated
    """
    data.current_stage = "upload"

    if not data.chunks:
        data.log_warning("No chunks to upload")
        data.uploaded_items = []
        return data

    if not data.item or not data.target_dataset:
        data.log_error("Missing item or target_dataset")
        return data

    chunk_metadata_list = data.chunk_metadata
    image_id_map = data.metadata.get('image_id_map', {})

    # Map image indices to actual image IDs
    if chunk_metadata_list and image_id_map:
        for chunk_meta in chunk_metadata_list:
            image_indices = chunk_meta.get('image_indices', [])
            actual_image_ids = [
                image_id_map[idx] for idx in image_indices if idx in image_id_map
            ]
            chunk_meta['image_ids'] = actual_image_ids

    if chunk_metadata_list and len(chunk_metadata_list) == len(data.chunks):
        uploaded_items = upload_chunks_bulk(
            chunks=data.chunks,
            chunk_metadata_list=chunk_metadata_list,
            source_item=data.item,
            target_dataset=data.target_dataset,
            processor_metadata=data.metadata,
        )
    else:
        uploaded_items = upload_chunks(
            chunks=data.chunks,
            source_item=data.item,
            target_dataset=data.target_dataset,
            remote_path='/chunks',
            processor_metadata=data.metadata,
        )

    data.uploaded_items = uploaded_items
    data.metadata['uploaded_count'] = len(uploaded_items)

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
        chunk_context = {
            **processor_metadata,
            **{
                k: v
                for k, v in chunk_meta.items()
                if k not in ['chunk_index', 'page_numbers', 'image_ids', 'image_indices']
            },
        }

        metadata = ChunkMetadata.create(
            source_item=source_item,
            total_chunks=len(chunks),
            chunk_index=idx,
            page_numbers=chunk_meta.get('page_numbers'),
            image_ids=chunk_meta.get('image_ids', []),
            processor=processor_metadata.get('processor'),
            extraction_method=processor_metadata.get('extraction_method'),
            processor_specific_metadata=chunk_context,
        ).to_dict()

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

    try:
        uploaded_items = target_dataset.items.upload_dataframe(
            df=df,
            item_metadata_column='metadata',
            remote_path_column='remote_path',
            local_path_column='text',
            file_name_column='filename',
            overwrite=True,
        )

        if uploaded_items is None:
            raise dl.PlatformException(f"No chunks were uploaded! Total chunks: {len(chunks)}")
        elif isinstance(uploaded_items, dl.Item):
            uploaded_items = [uploaded_items]
        else:
            uploaded_items = [item for item in uploaded_items]

        return uploaded_items
    except AttributeError:
        return upload_chunks(
            chunks=chunks,
            source_item=source_item,
            target_dataset=target_dataset,
            remote_path='/chunks',
            processor_metadata=processor_metadata,
        )


def upload_metadata_only(data) -> Any:
    """
    Upload only metadata without creating chunk items.

    Args:
        data: ExtractedData with item and metadata

    Returns:
        ExtractedData with metadata uploaded confirmation
    """
    data.current_stage = "metadata_upload"

    if not data.item:
        data.log_error("Missing item")
        return data

    try:
        data.item.metadata['user'] = data.item.metadata.get('user', {})
        data.item.metadata['user'].update(data.metadata)
        data.item.update(system_metadata=True)
        data.metadata['metadata_uploaded'] = True
    except Exception as e:
        data.log_warning(f"Failed to update metadata: {e}")
        data.metadata['metadata_uploaded'] = False

    return data


def dry_run_upload(data) -> Any:
    """
    Simulate upload without actually uploading.

    Args:
        data: ExtractedData with chunks

    Returns:
        ExtractedData with simulated upload info
    """
    data.current_stage = "dry_run"

    chunks = data.chunks or []
    data.uploaded_items = [f"simulated_item_{i}" for i in range(len(chunks))]
    data.metadata['dry_run'] = True
    data.metadata['uploaded_count'] = len(chunks)

    return data
