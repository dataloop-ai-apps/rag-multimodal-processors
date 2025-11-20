"""
Dataloop helper utilities for dataset management and chunk uploading.
Handles dataset creation, item upload, and metadata management.
"""

import shutil
import dtlpy as dl
import logging
import io
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from .chunk_metadata import ChunkMetadata

logger = logging.getLogger("rag-preprocessor")


def get_or_create_target_dataset(source_item: dl.Item, target_dataset: Optional[str]) -> dl.Dataset:
    """
    Get the target dataset for chunks or create one if needed.

    Args:
        source_item (dl.Item): Source item to get dataset context
        target_dataset (Optional[str]): Target dataset ID if specified

    Returns:
        dl.Dataset: Target dataset for chunks
    """
    logger.info(
        f"Determining target dataset | item_id={source_item.id} " f"target_dataset={target_dataset or 'auto'}"
    )

    if target_dataset:
        try:
            target_ds = dl.datasets.get(dataset_id=target_dataset)
            logger.info(f"Using specified target dataset | dataset_id={target_ds.id} " f"dataset_name={target_ds.name}")
            return target_ds
        except dl.exceptions.NotFound:
            logger.warning(f"Target dataset with ID {target_dataset} not found, a new dataset will be created")

    # Auto-create chunks dataset
    source_dataset_name = source_item.dataset.name
    new_dataset_name = f"{source_dataset_name}_chunks"
    project_id = source_item.project.id

    logger.info(
        f"Auto-creating chunks dataset | source_dataset={source_dataset_name} "
        f"new_dataset={new_dataset_name} project_id={project_id}"
    )

    try:
        # Check if dataset already exists
        target_ds = source_item.project.datasets.get(dataset_name=new_dataset_name)
        logger.info(f"Using existing chunks dataset | dataset_id={target_ds.id} " f"dataset_name={target_ds.name}")
    except dl.exceptions.NotFound:
        # Create new dataset
        target_ds = source_item.project.datasets.create(dataset_name=new_dataset_name)
        logger.info(f"Created new chunks dataset | dataset_id={target_ds.id} " f"dataset_name={target_ds.name}")

    return target_ds


def upload_chunks(
    chunks: List[str],
    source_item: dl.Item,
    target_dataset: dl.Dataset,
    remote_path: str,
    processor_metadata: Dict[str, Any],
    chunk_metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List[dl.Item]:
    """
    Upload text chunks as items to the target dataset using pandas DataFrame bulk upload.
    This approach allows individual metadata per chunk in a single API call.

    Args:
        chunks (List[str]): Text chunks to upload
        source_item (dl.Item): Source document item
        target_dataset (dl.Dataset): Target dataset for chunks
        remote_path (str): Remote path in dataset
        processor_metadata (Dict[str, Any]): Metadata specific to the processor
        chunk_metadata_list (Optional[List[Dict[str, Any]]]): Optional list of per-chunk metadata

    Returns:
        List[dl.Item]: Uploaded chunk items
    """
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
            # Use provided metadata (should already be a dict)
            chunk_meta = chunk_metadata_list[idx]
            # Ensure it's a dict and has the proper structure
            if isinstance(chunk_meta, dict):
                chunk_metadata = chunk_meta
            else:
                # If it's a ChunkMetadata object, convert to dict
                chunk_metadata = chunk_meta.to_dict() if hasattr(chunk_meta, 'to_dict') else chunk_meta
        else:
            # Create default metadata for this chunk
            chunk_metadata = ChunkMetadata.create(
                source_item=source_item,
                total_chunks=len(chunks),
                chunk_index=idx,  # Individual chunk index
                processor_specific_metadata=processor_metadata
            ).to_dict()  # Convert to dict for Dataloop API

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
        uploaded_items = target_dataset.items.upload(
            local_path=df,
            overwrite=True
        )

        # Handle response
        if uploaded_items is None:
            raise dl.PlatformException(f"Bulk upload returned None")
        elif isinstance(uploaded_items, dl.Item):
            uploaded_items = [uploaded_items]
        else:
            uploaded_items = list(uploaded_items)

        logger.info(f"Bulk upload successful | uploaded_count={len(uploaded_items)}")

    except Exception as e:
        logger.warning(f"Bulk upload failed: {str(e)}. Falling back to individual uploads...")

        # Fallback: Upload items one by one
        uploaded_items = []
        failed_uploads = []

        for idx, row in enumerate(upload_data):
            try:
                logger.debug(f"Uploading chunk {idx + 1}/{len(upload_data)}: {row['remote_name']}")

                # Upload single item
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
            raise dl.PlatformException(f"All individual uploads failed! Total chunks: {len(chunks)}")

        logger.info(f"Individual upload completed | successful={len(uploaded_items)}/{len(chunks)}")

    try:
        uploaded_names = [it.name for it in uploaded_items]
    except Exception:
        uploaded_names = ["<unknown>"]

    logger.info(
        f"Upload completed | item_id={source_item.id} uploaded_count={len(uploaded_items)} "
        f"remote_path={full_remote_path} sample_names={uploaded_names[:3]}"
    )

    return uploaded_items


def cleanup_temp_items_and_folder(
    items: List[dl.Item], folder_path: str, dataset: dl.Dataset, local_temp_dir: Optional[str] = None
) -> None:
    """
    Clean up temporary items, folder, and local directory.

    Args:
        items: List of Dataloop items to delete
        folder_path: Remote folder path in dataset to clean up
        dataset: Dataset containing the items
        local_temp_dir: Optional local temp directory to delete
    """
    # Delete uploaded items
    if items:
        logger.info(f"Cleaning up uploaded items | count={len(items)}")
        for item in items:
            try:
                item.delete()
                logger.debug(f"Deleted item | item_id={item.id}")
            except Exception as e:
                logger.warning(f"Failed to delete item {item.id}: {e}")

    # Delete temp folder in Dataloop
    try:
        logger.info(f"Deleting temp folder in Dataloop | folder={folder_path}")
        filters = dl.Filters()
        filters.add(field='dir', values=folder_path)
        remaining_items = dataset.items.list(filters=filters)
        for item in remaining_items:
            try:
                item.delete()
            except:
                pass
        logger.info("Temp folder cleanup completed")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp folder: {e}")

    # Delete local temp directory
    if local_temp_dir:
        try:
            shutil.rmtree(local_temp_dir, ignore_errors=True)
            logger.debug(f"Deleted local temp directory | path={local_temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete local temp dir: {e}")
