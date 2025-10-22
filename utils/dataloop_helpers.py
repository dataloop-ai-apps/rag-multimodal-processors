"""
Dataloop helper utilities for dataset management and chunk uploading.
Handles dataset creation, item upload, and metadata management.
"""

import dtlpy as dl
import logging
import io
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger('item-processor-logger')


def get_or_create_target_dataset(original_item: dl.Item, target_dataset: Optional[str]) -> dl.Dataset:
    """
    Get the target dataset for chunks or create one if needed.
    
    Args:
        original_item (dl.Item): Original item to get dataset context
        target_dataset (Optional[str]): Target dataset ID if specified
        
    Returns:
        dl.Dataset: Target dataset for chunks
    """
    logger.info(
        f"Determining target dataset | item_id={original_item.id} "
        f"target_dataset={target_dataset or 'auto'}"
    )
    
    if target_dataset:
        try:
            target_ds = dl.datasets.get(dataset_id=target_dataset)
            logger.info(
                f"Using specified target dataset | dataset_id={target_ds.id} "
                f"dataset_name={target_ds.name}"
            )
            return target_ds
        except dl.exceptions.NotFound:
            logger.warning(f"Target dataset with ID {target_dataset} not found, a new dataset will be created")
    
    # Auto-create chunks dataset
    original_dataset_name = original_item.dataset.name
    new_dataset_name = f"{original_dataset_name}_chunks"
    project_id = original_item.project.id
    
    logger.info(
        f"Auto-creating chunks dataset | original_dataset={original_dataset_name} "
        f"new_dataset={new_dataset_name} project_id={project_id}"
    )
    
    try:
        # Check if dataset already exists
        target_ds = original_item.project.datasets.get(dataset_name=new_dataset_name)
        logger.info(
            f"Using existing chunks dataset | dataset_id={target_ds.id} "
            f"dataset_name={target_ds.name}"
        )
    except dl.exceptions.NotFound:
        # Create new dataset
        target_ds = original_item.project.datasets.create(dataset_name=new_dataset_name)
        logger.info(
            f"Created new chunks dataset | dataset_id={target_ds.id} "
            f"dataset_name={target_ds.name}"
        )
    
    return target_ds


def upload_chunks(chunks: List[str], 
                 original_item: dl.Item, 
                 target_dataset: dl.Dataset,
                 remote_path: str,
                 processor_metadata: Dict[str, Any]) -> List[dl.Item]:
    """
    Upload text chunks as items to the target dataset using BytesIO buffers.
    
    Args:
        chunks (List[str]): Text chunks to upload
        original_item (dl.Item): Original document item
        target_dataset (dl.Dataset): Target dataset for chunks
        remote_path (str): Remote path in dataset
        processor_metadata (Dict[str, Any]): Metadata specific to the processor
        
    Returns:
        List[dl.Item]: Uploaded chunk items
    """
    logger.info(
        f"Uploading chunks | item_id={original_item.id} count={len(chunks)} "
        f"remote_path={remote_path} target_dataset={target_dataset.name}"
    )
    
    # Create BytesIO buffers for each chunk
    binaries = []
    for idx, chunk in enumerate(chunks):
        base_name = Path(original_item.name).stem
        chunk_filename = f"{base_name}_chunk_{idx:04d}.txt"
        
        # Create BytesIO buffer
        buffer = io.BytesIO(chunk.encode('utf-8'))
        buffer.name = chunk_filename
        buffer.seek(0)
        binaries.append(buffer)
    
    # Create metadata
    base_metadata = create_chunk_metadata(
        original_item=original_item,
        total_chunks=len(chunks),
        processor_metadata=processor_metadata
    )
    
    full_remote_path = os.path.join(remote_path, original_item.dir.lstrip('/')).replace('\\', '/')
    
    # Bulk upload chunks
    uploaded_items = target_dataset.items.upload(
        local_path=binaries,
        remote_path=full_remote_path,
        item_metadata=base_metadata,
        overwrite=True,
        raise_on_error=True,
    )
    
    # Handle single item vs list response
    if uploaded_items is None:
        logger.error(
            f"Upload returned None | item_id={original_item.id} chunks={len(binaries)}"
        )
        raise dl.PlatformException(f"No chunks were uploaded! Total chunks: {len(binaries)}")
    elif isinstance(uploaded_items, dl.Item):
        chunk_items = [uploaded_items]
    else:
        chunk_items = [item for item in uploaded_items]
    
    try:
        uploaded_names = [it.name for it in chunk_items]
    except Exception:
        uploaded_names = ["<unknown>"]
    
    logger.info(
        f"Upload completed | item_id={original_item.id} uploaded_count={len(chunk_items)} "
        f"remote_path={full_remote_path} sample_names={uploaded_names[:3]}"
    )
    
    return chunk_items


def create_chunk_metadata(original_item: dl.Item,
                         total_chunks: int,
                         processor_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create standardized metadata for chunk items.
    
    Args:
        original_item (dl.Item): Original document item
        total_chunks (int): Total number of chunks
        processor_metadata (Dict[str, Any]): Processor-specific metadata
        
    Returns:
        Dict[str, Any]: Standardized metadata structure
    """
    metadata = {
        'user': {
            'document': original_item.name,
            'document_type': original_item.mimetype,
            'total_chunks': total_chunks,
            'extracted_chunk': True,
            'original_item_id': original_item.id,
            'original_dataset_id': original_item.dataset.id,
            'processing_timestamp': time.time()
        }
    }
    
    # Merge processor-specific metadata
    metadata['user'].update(processor_metadata)
    
    return metadata


def cleanup_temp_items_and_folder(
    items: List[dl.Item],
    folder_path: str,
    dataset: dl.Dataset,
    local_temp_dir: Optional[str] = None
) -> None:
    """
    Clean up temporary items, folder, and local directory.
    
    Args:
        items: List of Dataloop items to delete
        folder_path: Remote folder path in dataset to clean up
        dataset: Dataset containing the items
        local_temp_dir: Optional local temp directory to delete
    """
    import shutil
    
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

