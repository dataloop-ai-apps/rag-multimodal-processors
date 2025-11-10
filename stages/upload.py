"""
Upload stages for storing chunks in Dataloop.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any, List
import dtlpy as dl


def upload_to_dataloop(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload chunks to Dataloop dataset.

    Args:
        data: Must contain 'chunks', 'item', 'target_dataset'
        config: Not used

    Returns:
        data with 'uploaded_items' added
    """
    try:
        from utils.dataloop_helpers import upload_chunks
    except ImportError:
        print("Warning: dataloop_helpers not found, cannot upload chunks")
        data['uploaded_items'] = []
        return data

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

    # Upload chunks
    uploaded_items = upload_chunks(
        chunks=chunks,
        original_item=item,
        target_dataset=target_dataset,
        remote_path='/chunks',
        processor_metadata=metadata,
    )

    data['uploaded_items'] = uploaded_items
    data['metadata']['uploaded_count'] = len(uploaded_items)

    return data


def upload_with_images(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload chunks along with extracted images.

    Args:
        data: Must contain 'chunks', 'images', 'item', 'target_dataset'
        config: Can contain:
            - 'upload_images' (bool): Whether to upload images (default: True)
            - 'image_upload_path' (str): Remote path for images (default: '/images')

    Returns:
        data with 'uploaded_items' and 'uploaded_images' added
    """
    # First upload chunks
    data = upload_to_dataloop(data, config)

    # Optionally upload images
    if not config.get('upload_images', True):
        return data

    images = data.get('images', [])
    target_dataset = data.get('target_dataset')
    item = data.get('item')

    if not images or not target_dataset:
        return data

    try:
        from utils.dataloop_helpers import upload_images
    except ImportError:
        print("Warning: dataloop_helpers.upload_images not found, using fallback")
        # Fallback to simple upload
        uploaded_images = []
        image_upload_path = config.get('image_upload_path', '/images')
        for img in images:
            img_path = img.get('path') if isinstance(img, dict) else (img.path if hasattr(img, 'path') else None)
            if not img_path:
                continue
            try:
                uploaded_img = target_dataset.items.upload(local_path=img_path, remote_path=image_upload_path)
                uploaded_images.append(uploaded_img)
            except Exception as e:
                print(f"Warning: Failed to upload image {img_path}: {e}")
        data['uploaded_images'] = uploaded_images
        data['metadata']['uploaded_image_count'] = len(uploaded_images)
        return data

    # Use helper function for proper image upload with metadata
    image_upload_path = config.get('image_upload_path', '/images')
    uploaded_images = upload_images(
        images=images, original_item=item, target_dataset=target_dataset, remote_path=image_upload_path
    )

    data['uploaded_images'] = uploaded_images
    data['metadata']['uploaded_image_count'] = len(uploaded_images)

    return data


def upload_with_metadata_only(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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

    except Exception as e:
        print(f"Warning: Failed to update metadata: {e}")
        data['metadata_uploaded'] = False

    return data


def dry_run_upload(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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
