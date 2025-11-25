"""
Upload transforms for storing chunks in Dataloop.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

from .dataloop_helpers import upload_chunks
from .extracted_data import ExtractedData


def upload_to_dataloop(data: ExtractedData) -> ExtractedData:
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

    uploaded_items = upload_chunks(
        chunks=data.chunks,
        source_item=data.item,
        target_dataset=data.target_dataset,
        remote_path='/chunks',
        processor_metadata=data.metadata,
        chunk_metadata_list=chunk_metadata_list if chunk_metadata_list else None,
    )

    data.uploaded_items = uploaded_items
    data.metadata['uploaded_count'] = len(uploaded_items)

    return data


def upload_metadata_only(data: ExtractedData) -> ExtractedData:
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


def dry_run_upload(data: ExtractedData) -> ExtractedData:
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
