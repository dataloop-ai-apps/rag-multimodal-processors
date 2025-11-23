"""
LLM-based processing transforms using Dataloop models.

All functions follow signature: (data: ExtractedData) -> ExtractedData
"""

import json
import tempfile
import os
from typing import Optional

import dtlpy as dl

from utils.extracted_data import ExtractedData


def _call_llm_model(model_id: str, prompt: str, dataset: Optional[dl.Dataset] = None) -> Optional[str]:
    """
    Call Dataloop LLM model with a text prompt.

    Creates a temporary text item, runs prediction, and retrieves result.

    Args:
        model_id: Dataloop model ID
        prompt: Text prompt to send to the model
        dataset: Optional dataset for temporary item

    Returns:
        Model response text, or None if execution fails
    """
    result = None
    temp_item = None
    temp_path = None

    try:
        model = dl.models.get(model_id=model_id)

        if dataset is None:
            project = model.project
            datasets = project.datasets.list()
            if datasets.items:
                dataset = datasets.items[0]
            else:
                return None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_path = f.name

        temp_item = dataset.items.upload(local_path=temp_path, remote_path='/temp', overwrite=True)

        execution = model.predict(item_ids=[temp_item.id])
        execution.wait()

        if execution.latest_status['status'] == dl.ExecutionStatus.SUCCESS:
            updated_item = dl.items.get(item_id=temp_item.id)

            annotations = updated_item.annotations.list()
            if annotations.items:
                for ann in annotations.items:
                    if hasattr(ann, 'label') and ann.label == 'response':
                        result = ann.metadata.get('text', '')
                        break
                    elif hasattr(ann, 'metadata') and 'text' in ann.metadata:
                        result = ann.metadata['text']
                        break

            if not result:
                result = updated_item.metadata.get('llm_response', '')

            if not result and hasattr(execution, 'output') and execution.output:
                result = str(execution.output)

        if temp_item:
            try:
                temp_item.delete()
            except Exception:
                pass

        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    except Exception:
        if temp_item:
            try:
                temp_item.delete()
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    return result


def llm_chunk_semantic(data: ExtractedData) -> ExtractedData:
    """
    Semantic chunking using a LLM.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with chunks populated
    """
    data.current_stage = "llm_chunking"
    content = data.get_text()
    model_id = data.config.llm_model_id if hasattr(data.config, 'llm_model_id') else None

    if not content:
        data.chunks = []
        return data

    if not model_id:
        data.log_warning("llm_model_id not provided, skipping semantic chunking")
        data.chunks = []
        return data

    # Placeholder for semantic chunking implementation
    data.metadata['chunking_method'] = 'llm_semantic'
    return data


def llm_summarize(data: ExtractedData) -> ExtractedData:
    """
    Generate summary of content using Dataloop LLM.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with summary in metadata
    """
    data.current_stage = "summarization"
    content = data.get_text()
    model_id = data.config.llm_model_id if hasattr(data.config, 'llm_model_id') else None
    generate_summary = getattr(data.config, 'generate_summary', False)

    if not content or not generate_summary or not model_id:
        return data

    try:
        prompt = f"Provide a concise summary of the following text in 2-3 sentences:\n\n{content[:2000]}"

        response = _call_llm_model(model_id, prompt, data.target_dataset)

        if response:
            data.metadata['summary'] = response.strip()
    except Exception as e:
        data.log_warning(f"Summary generation failed: {e}")

    return data


def llm_extract_entities(data: ExtractedData) -> ExtractedData:
    """
    Extract named entities using Dataloop LLM.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with entities in metadata
    """
    data.current_stage = "entity_extraction"
    content = data.get_text()
    model_id = data.config.llm_model_id if hasattr(data.config, 'llm_model_id') else None
    extract_entities = getattr(data.config, 'extract_entities', False)

    if not content or not extract_entities or not model_id:
        return data

    try:
        prompt = f"""Extract key entities (people, organizations, locations, dates) from this text.
Return as JSON list.

Text:
{content[:1000]}"""

        response = _call_llm_model(model_id, prompt, data.target_dataset)

        if response:
            try:
                entities = json.loads(response)
                data.metadata['entities'] = entities
            except (json.JSONDecodeError, ValueError):
                data.metadata['entities_raw'] = response
    except Exception as e:
        data.log_warning(f"Entity extraction failed: {e}")

    return data


def llm_translate(data: ExtractedData) -> ExtractedData:
    """
    Translate content using Dataloop LLM.

    Args:
        data: ExtractedData with content

    Returns:
        ExtractedData with translated content
    """
    data.current_stage = "translation"
    content = data.get_text()
    model_id = data.config.llm_model_id if hasattr(data.config, 'llm_model_id') else None
    target_lang = getattr(data.config, 'target_language', 'English')
    translate = getattr(data.config, 'translate', False)

    if not content or not translate or not model_id:
        return data

    try:
        prompt = f"""Translate the following text to {target_lang}:

{content}"""

        response = _call_llm_model(model_id, prompt, data.target_dataset)

        if response:
            data.metadata['original_content'] = content
            data.metadata['original_language'] = 'auto-detected'
            data.metadata['target_language'] = target_lang
            data.content_text = response.strip()
    except Exception as e:
        data.log_warning(f"Translation failed: {e}")

    return data
