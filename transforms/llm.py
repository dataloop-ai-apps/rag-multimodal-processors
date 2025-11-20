"""
LLM-based processing stages using Dataloop models.
All stage functions follow signature: (data: dict, config: dict) -> dict
"""

import json
import tempfile
import os
import dtlpy as dl
from typing import Dict, Any, Optional


def _call_llm_model(model_id: str, prompt: str, dataset: Optional[dl.Dataset] = None) -> Optional[str]:
    """
    Helper function to call Dataloop LLM model with a text prompt.

    Creates a temporary text item, runs prediction, and retrieves result.

    Args:
        model_id: Dataloop model ID
        prompt: Text prompt to send to the model
        dataset: Optional dataset to create temporary item in (uses model's dataset if not provided)

    Returns:
        Model response text, or None if execution fails
    """
    result = None
    temp_item = None
    temp_path = None

    try:
        # Get model
        model = dl.models.get(model_id=model_id)

        # Get dataset for temporary item
        if dataset is None:
            # Try to get dataset from model's project
            project = model.project
            datasets = project.datasets.list()
            if datasets.items:
                dataset = datasets.items[0]
            else:
                print("Warning: No dataset available for temporary item creation")
                return None

        # Create temporary text file with prompt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_path = f.name

        # Upload as temporary item
        temp_item = dataset.items.upload(local_path=temp_path, remote_path='/temp', overwrite=True)

        # Execute model prediction
        execution = model.predict(item_ids=[temp_item.id])
        execution.wait()

        # Check execution status
        if execution.latest_status['status'] == dl.ExecutionStatus.SUCCESS:
            # Get result from execution or item annotations
            # For LLM models, result is typically in item annotations or execution output
            updated_item = dl.items.get(item_id=temp_item.id)

            # Try to get result from annotations (common for LLM models)
            annotations = updated_item.annotations.list()
            if annotations.items:
                # Get text from first annotation or annotation field
                for ann in annotations.items:
                    if hasattr(ann, 'label') and ann.label == 'response':
                        result = ann.metadata.get('text', '')
                        break
                    elif hasattr(ann, 'metadata') and 'text' in ann.metadata:
                        result = ann.metadata['text']
                        break

            # If no annotation, try to get from item metadata or execution output
            if not result:
                result = updated_item.metadata.get('llm_response', '')

            # If still no result, check execution output
            if not result and hasattr(execution, 'output') and execution.output:
                result = str(execution.output)

        # Clean up temporary item
        if temp_item:
            try:
                temp_item.delete()
            except Exception:
                pass

        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: LLM model call failed: {e}")
        # Clean up on error
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


def llm_chunk_semantic(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Semantic chunking using a LLM.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id'
            Optional: 'prompt_chunk' for custom prompt (falls back to default if not provided)

    Returns:
        data with 'chunks' list added
    """
    content = data.get('content', '')
    model_id = config.get('llm_model_id')

    if not content:
        data['chunks'] = []
    elif not model_id:
        print("Warning: llm_model_id not provided, skipping semantic chunking")
        data['chunks'] = []
    else:
        # TODO: Implement actual semantic chunking with LLM
        # For now, this is a placeholder that supports custom prompts from config
        # prompt_text = config.get('prompt_chunk')  # Reserved for future implementation
        data['chunks'] = data.get('chunks', [])
        data.setdefault('metadata', {})['chunking_method'] = 'llm_semantic'

    return data


def llm_summarize(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary of content using Dataloop LLM.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id'
            Optional: 'prompt_summarize' for custom prompt (falls back to default if not provided)

    Returns:
        data with 'summary' added to metadata
    """
    content = data.get('content', '')
    model_id = config.get('llm_model_id')
    generate_summary = config.get('generate_summary', False)

    if content and generate_summary and model_id:
        try:
            # Get prompt from config, fall back to default
            prompt_text = config.get('prompt_summarize')

            if prompt_text:
                # Use user-defined prompt, replacing {content} placeholder if present
                prompt = prompt_text.replace('{content}', content[:2000])
            else:
                # Default prompt
                prompt = f"Provide a concise summary of the following text in 2-3 sentences:\n\n{content[:2000]}"

            # Get dataset from data if available (for temporary item creation)
            dataset = data.get('target_dataset')
            response = _call_llm_model(model_id, prompt, dataset)

            if response:
                summary = response.strip()
                data.setdefault('metadata', {})['summary'] = summary
                print(f"Generated summary: {summary[:100]}...")
        except Exception as e:
            print(f"Warning: Summary generation failed: {e}")

    return data


def llm_extract_entities(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract named entities using Dataloop LLM.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id'
            Optional: 'prompt_entities' for custom prompt (falls back to default if not provided)

    Returns:
        data with 'entities' added to metadata
    """
    content = data.get('content', '')
    model_id = config.get('llm_model_id')
    extract_entities = config.get('extract_entities', False)

    if content and extract_entities and model_id:
        try:
            # Get prompt from config, fall back to default
            prompt_text = config.get('prompt_entities')

            if prompt_text:
                # Use user-defined prompt, replacing {content} placeholder if present
                prompt = prompt_text.replace('{content}', content[:1000])
            else:
                # Default prompt
                prompt = f"""Extract key entities (people, organizations, locations, dates) from this text.
Return as JSON list.

Text:
{content[:1000]}"""

            # Get dataset from data if available (for temporary item creation)
            dataset = data.get('target_dataset')
            response = _call_llm_model(model_id, prompt, dataset)

            if response:
                try:
                    entities = json.loads(response)
                    data.setdefault('metadata', {})['entities'] = entities
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, store as string
                    data.setdefault('metadata', {})['entities_raw'] = response
        except Exception as e:
            print(f"Warning: Entity extraction failed: {e}")

    return data


def llm_translate(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate content using Dataloop LLM.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id' and 'target_language'
            Optional: 'prompt_translate' for custom prompt (falls back to default if not provided)

    Returns:
        data with translated content
    """
    content = data.get('content', '')
    model_id = config.get('llm_model_id')
    target_lang = config.get('target_language', 'English')
    translate = config.get('translate', False)

    if content and translate and model_id:
        try:
            # Get prompt from config, fall back to default
            prompt_text = config.get('prompt_translate')

            if prompt_text:
                # Use user-defined prompt, replacing {content} and {target_language} placeholders if present
                prompt = prompt_text.replace('{content}', content).replace('{target_language}', target_lang)
            else:
                # Default prompt
                prompt = f"""Translate the following text to {target_lang}:

{content}"""

            # Get dataset from data if available (for temporary item creation)
            dataset = data.get('target_dataset')
            response = _call_llm_model(model_id, prompt, dataset)

            if response:
                translated = response.strip()

                # Store original in metadata
                data.setdefault('metadata', {})['original_content'] = content
                data.setdefault('metadata', {})['original_language'] = 'auto-detected'
                data.setdefault('metadata', {})['target_language'] = target_lang

                # Replace content with translation
                data['content'] = translated
        except Exception as e:
            print(f"Warning: Translation failed: {e}")

    return data
