"""
LLM-based processing stages using Dataloop models.
All stage functions follow signature: (data: dict, config: dict) -> dict
"""

import dtlpy as dl
from typing import Dict, Any


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

    if not content:
        data['chunks'] = []
        return data

    model_id = config.get('llm_model_id')
    if not model_id:
        print("Warning: llm_model_id not provided, skipping semantic chunking")
        return data

    # TODO: Implement actual semantic chunking with LLM
    # For now, this is a placeholder that supports custom prompts from config
    # prompt_text = config.get('prompt_chunk')  # Reserved for future implementation

    data['chunks'] = data['content']
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

    if not content or not config.get('generate_summary', False):
        return data

    model_id = config.get('llm_model_id')
    if not model_id:
        print("Warning: llm_model_id not provided, skipping summary generation")
        return data

    try:
        model = dl.models.get(model_id=model_id)

        # Get prompt from config, fall back to default
        prompt_text = config.get('prompt_summarize')

        if prompt_text:
            # Use user-defined prompt, replacing {content} placeholder if present
            prompt = prompt_text.replace('{content}', content[:2000])
        else:
            # Default prompt
            prompt = f"""Provide a concise summary of the following text in 2-3 sentences:

{content[:2000]}"""  # Limit to first 2000 chars

        response = model.predict([prompt])

        if response and len(response) > 0:
            summary = response[0].strip()
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

    if not content or not config.get('extract_entities', False):
        return data

    model_id = config.get('llm_model_id')
    if not model_id:
        return data

    try:
        model = dl.models.get(model_id=model_id)

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

        response = model.predict([prompt])

        if response and len(response) > 0:
            import json

            try:
                entities = json.loads(response[0])
                data.setdefault('metadata', {})['entities'] = entities
            except:
                # If not valid JSON, store as string
                data.setdefault('metadata', {})['entities_raw'] = response[0]

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

    if not content or not config.get('translate', False):
        return data

    model_id = config.get('llm_model_id')
    target_lang = config.get('target_language', 'English')

    if not model_id:
        print("Warning: llm_model_id not provided, skipping translation")
        return data

    try:
        model = dl.models.get(model_id=model_id)

        # Get prompt from config, fall back to default
        prompt_text = config.get('prompt_translate')

        if prompt_text:
            # Use user-defined prompt, replacing {content} and {target_language} placeholders if present
            prompt = prompt_text.replace('{content}', content).replace('{target_language}', target_lang)
        else:
            # Default prompt
            prompt = f"""Translate the following text to {target_lang}:

{content}"""

        response = model.predict([prompt])

        if response and len(response) > 0:
            translated = response[0].strip()

            # Store original in metadata
            data.setdefault('metadata', {})['original_content'] = content
            data.setdefault('metadata', {})['original_language'] = 'auto-detected'
            data.setdefault('metadata', {})['target_language'] = target_lang

            # Replace content with translation
            data['content'] = translated

    except Exception as e:
        print(f"Warning: Translation failed: {e}")

    return data
