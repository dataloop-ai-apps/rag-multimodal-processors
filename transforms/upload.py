"""
Upload transforms for storing chunks in Dataloop.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

from typing import List, Dict, Any, Optional

import dtlpy as dl

from utils.dataloop_helpers import upload_chunks
from utils.extracted_data import ExtractedData


class ChunkUploader:
    """Chunk upload operations to Dataloop."""

    @staticmethod
    def upload(
        chunks: List[str],
        source_item: dl.Item,
        target_dataset: dl.Dataset,
        remote_path: str = '/chunks',
        processor_metadata: Optional[Dict[str, Any]] = None,
        chunk_metadata_list: Optional[List[Dict[str, Any]]] = None,
        image_id_map: Optional[Dict[int, str]] = None
    ) -> List[dl.Item]:
        """Upload chunks to Dataloop dataset."""
        if chunk_metadata_list and image_id_map:
            for chunk_meta in chunk_metadata_list:
                image_indices = chunk_meta.get('image_indices', [])
                actual_image_ids = [
                    image_id_map[idx] for idx in image_indices if idx in image_id_map
                ]
                chunk_meta['image_ids'] = actual_image_ids

        return upload_chunks(
            chunks=chunks,
            source_item=source_item,
            target_dataset=target_dataset,
            remote_path=remote_path,
            processor_metadata=processor_metadata,
            chunk_metadata_list=chunk_metadata_list,
        )

    @staticmethod
    def upload_metadata(item: dl.Item, metadata: Dict[str, Any]) -> bool:
        """Upload metadata to an existing Dataloop item."""
        try:
            item.metadata['user'] = item.metadata.get('user', {})
            item.metadata['user'].update(metadata)
            item.update(system_metadata=True)
            return True
        except Exception:
            return False

    @staticmethod
    def simulate_upload(chunks: List[str]) -> List[str]:
        """Simulate upload for dry-run testing."""
        return [f"simulated_item_{i}" for i in range(len(chunks))]


# Transform wrappers

def upload_to_dataloop(data: ExtractedData) -> ExtractedData:
    """Upload chunks to Dataloop dataset with optional image associations."""
    data.current_stage = "upload"

    if not data.chunks:
        data.log_warning("No chunks to upload")
        data.uploaded_items = []
        return data

    if not data.item or not data.target_dataset:
        data.log_error("Missing source item or target dataset.")
        return data

    uploaded_items = ChunkUploader.upload(
        chunks=data.chunks,
        source_item=data.item,
        target_dataset=data.target_dataset,
        remote_path='/chunks',
        processor_metadata=data.metadata,
        chunk_metadata_list=data.chunk_metadata if data.chunk_metadata else None,
        image_id_map=data.metadata.get('image_id_map', {}),
    )

    data.uploaded_items = uploaded_items
    data.metadata['uploaded_count'] = len(uploaded_items)

    return data


def upload_metadata_only(data: ExtractedData) -> ExtractedData:
    """Upload only metadata without creating chunk items."""
    data.current_stage = "metadata_upload"

    if not data.item:
        data.log_error("Missing source item.")
        return data

    success = ChunkUploader.upload_metadata(data.item, data.metadata)
    data.metadata['metadata_uploaded'] = success

    if not success:
        data.log_warning("Failed to update metadata")

    return data


def dry_run_upload(data: ExtractedData) -> ExtractedData:
    """Simulate upload without actually uploading."""
    data.current_stage = "dry_run"

    chunks = data.chunks or []
    data.uploaded_items = ChunkUploader.simulate_upload(chunks)
    data.metadata['dry_run'] = True
    data.metadata['uploaded_count'] = len(chunks)

    return data
