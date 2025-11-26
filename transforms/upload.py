"""
Upload transforms for storing chunks in Dataloop.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import io
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import dtlpy as dl
import pandas as pd

from utils.chunk_metadata import ChunkMetadata
from utils.extracted_data import ExtractedData

logger = logging.getLogger("rag-preprocessor")


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
        """Upload chunks to Dataloop dataset using pandas DataFrame bulk upload."""
        # Resolve image IDs if provided
        if chunk_metadata_list and image_id_map:
            for chunk_meta in chunk_metadata_list:
                image_indices = chunk_meta.get('image_indices', [])
                actual_image_ids = [
                    image_id_map[idx] for idx in image_indices if idx in image_id_map
                ]
                chunk_meta['image_ids'] = actual_image_ids

        logger.info(
            f"Uploading chunks | item_id={source_item.id} count={len(chunks)} "
            f"remote_path={remote_path} target_dataset={target_dataset.name}"
        )

        # Prepare DataFrame with individual metadata for each chunk
        upload_data = []
        base_name = Path(source_item.name).stem
        full_remote_path = os.path.join(remote_path, source_item.dir.lstrip('/')).replace('\\', '/')

        for idx, chunk_text in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk_{idx:04d}.txt"

            # Create BytesIO buffer for the chunk
            buffer = io.BytesIO(chunk_text.encode('utf-8'))
            buffer.name = chunk_filename
            buffer.seek(0)

            # Use provided metadata if available, otherwise create default
            if chunk_metadata_list and idx < len(chunk_metadata_list):
                chunk_meta = chunk_metadata_list[idx]
                if isinstance(chunk_meta, dict):
                    chunk_metadata = chunk_meta
                else:
                    chunk_metadata = chunk_meta.to_dict() if hasattr(chunk_meta, 'to_dict') else chunk_meta
            else:
                chunk_metadata = ChunkMetadata.create(
                    source_item=source_item,
                    total_chunks=len(chunks),
                    chunk_index=idx,
                    processor_specific_metadata=processor_metadata
                ).to_dict()

            upload_data.append({
                'local_path': buffer,
                'remote_path': full_remote_path,
                'remote_name': chunk_filename,
                'item_metadata': chunk_metadata
            })

        # Create DataFrame for bulk upload
        df = pd.DataFrame(upload_data)

        # Try bulk upload first
        try:
            logger.info("Attempting bulk upload with pandas DataFrame...")
            uploaded_items = target_dataset.items.upload(local_path=df, overwrite=True)

            if uploaded_items is None:
                raise dl.PlatformException("Bulk upload returned None")
            elif isinstance(uploaded_items, dl.Item):
                uploaded_items = [uploaded_items]
            else:
                uploaded_items = list(uploaded_items)

            logger.info(f"Bulk upload successful | uploaded_count={len(uploaded_items)}")

        except Exception as e:
            logger.warning(f"Bulk upload failed: {str(e)}. Falling back to individual uploads...")
            uploaded_items = ChunkUploader._fallback_individual_upload(
                upload_data, target_dataset, len(chunks)
            )

        try:
            uploaded_names = [it.name for it in uploaded_items]
        except Exception:
            uploaded_names = ["<unknown>"]

        logger.info(
            f"Upload completed | item_id={source_item.id} uploaded_count={len(uploaded_items)} "
            f"remote_path={full_remote_path} sample_names={uploaded_names[:3]}"
        )

        return uploaded_items

    @staticmethod
    def _fallback_individual_upload(
        upload_data: List[Dict],
        target_dataset: dl.Dataset,
        total_chunks: int
    ) -> List[dl.Item]:
        """Fallback to individual uploads when bulk upload fails."""
        uploaded_items = []
        failed_uploads = []

        for idx, row in enumerate(upload_data):
            try:
                logger.debug(f"Uploading chunk {idx + 1}/{len(upload_data)}: {row['remote_name']}")

                uploaded_item = target_dataset.items.upload(
                    local_path=row['local_path'],
                    remote_path=row['remote_path'],
                    remote_name=row['remote_name'],
                    item_metadata=row['item_metadata'],
                    overwrite=True
                )

                if uploaded_item:
                    uploaded_items.append(uploaded_item)
                else:
                    failed_uploads.append(idx)

            except Exception as item_e:
                logger.error(f"Failed to upload chunk {idx}: {str(item_e)}")
                failed_uploads.append(idx)

        if failed_uploads:
            logger.error(f"Failed to upload {len(failed_uploads)} chunks: indices {failed_uploads}")

        if not uploaded_items:
            raise dl.PlatformException(f"All individual uploads failed! Total chunks: {total_chunks}")

        logger.info(f"Individual upload completed | successful={len(uploaded_items)}/{total_chunks}")
        return uploaded_items

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
