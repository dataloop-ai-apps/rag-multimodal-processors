"""
LLM-based processing stages using Dataloop models.
All functions follow signature: (data: dict, config: dict) -> dict
"""

from typing import Dict, Any, List


def llm_chunk_semantic(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Semantic chunking using Dataloop LLM model.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id'

    Returns:
        data with 'chunks' list added
    """
    content = data.get('content', '')

    if not content:
        data['chunks'] = []
        return data

    model_id = config.get('llm_model_id')
    if not model_id:
        print("Warning: llm_model_id not provided, falling back to recursive chunking")
        from stages.chunking import chunk_recursive
        return chunk_recursive(data, config)

    import dtlpy as dl

    try:
        model = dl.models.get(model_id=model_id)

        # Prepare prompt for semantic chunking
        max_chunk_size = config.get('max_chunk_size', 300)
        prompt = f"""Split this text into semantic chunks. Each chunk should be self-contained and meaningful.
Maximum chunk size: {max_chunk_size} words.

Text:
{content}

Return chunks separated by '---CHUNK---'"""

        # Execute model
        response = model.predict([prompt])

        # Parse response
        if response and len(response) > 0:
            chunks = response[0].split('---CHUNK---')
            chunks = [c.strip() for c in chunks if c.strip()]
        else:
            chunks = [content]  # Fallback

        data['chunks'] = chunks
        data.setdefault('metadata', {})['chunking_method'] = 'llm_semantic'
        data['metadata']['chunk_count'] = len(chunks)
        data['metadata']['llm_model_used'] = model_id

    except Exception as e:
        print(f"Warning: LLM chunking failed ({e}), falling back to recursive")
        from stages.chunking import chunk_recursive
        return chunk_recursive(data, config)

    return data


def llm_summarize(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary of content using Dataloop LLM.

    Args:
        data: Must contain 'content' key
        config: Must contain 'llm_model_id'

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

    import dtlpy as dl

    try:
        model = dl.models.get(model_id=model_id)

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

    Returns:
        data with 'entities' added to metadata
    """
    content = data.get('content', '')

    if not content or not config.get('extract_entities', False):
        return data

    model_id = config.get('llm_model_id')
    if not model_id:
        return data

    import dtlpy as dl

    try:
        model = dl.models.get(model_id=model_id)

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

    import dtlpy as dl

    try:
        model = dl.models.get(model_id=model_id)

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
