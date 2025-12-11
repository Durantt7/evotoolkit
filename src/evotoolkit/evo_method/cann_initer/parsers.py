# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter common parsing utilities."""

import json
import re


def parse_json(response: str) -> dict:
    """Parse JSON from LLM response."""
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def parse_code(response: str, lang: str = None) -> str:
    """Parse code block from LLM response."""
    if lang:
        pattern = rf"```{lang}\s*(.*?)\s*```"
    else:
        pattern = r"```(?:cpp|c\+\+|python)?\s*(.*?)\s*```"
    code_match = re.search(pattern, response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response.strip()


def parse_tag(response: str, tag: str) -> str:
    """Parse content from <tag>...</tag> block.

    Args:
        response: LLM response containing tagged content
        tag: Tag name without angle brackets (e.g., "response", "init_body")

    Returns:
        Content inside the tag, stripped of leading/trailing whitespace.
        Returns empty string if tag not found.

    Example:
        >>> parse_tag("<response>hello</response>", "response")
        'hello'
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_multiple_tags(response: str, tags: list) -> dict:
    """Parse content from multiple <tag>...</tag> blocks.

    Args:
        response: LLM response containing tagged content
        tags: List of tag names to extract

    Returns:
        Dictionary mapping tag names to their content.
        Missing tags will have empty string values.

    Example:
        >>> parse_multiple_tags("<a>1</a><b>2</b>", ["a", "b", "c"])
        {'a': '1', 'b': '2', 'c': ''}
    """
    return {tag: parse_tag(response, tag) for tag in tags}
