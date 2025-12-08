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
