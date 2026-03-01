"""Utilities for extracting JSON from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


class JsonExtractError(ValueError):
    pass


_CODE_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text.

    Strategy:
    - Prefer a ```json codeblock.
    - Else take the first '{' .. last '}' window.
    """

    text = (text or "").strip()
    if not text:
        raise JsonExtractError("Empty response")

    m = _CODE_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
        except Exception as e:
            raise JsonExtractError("Invalid JSON in codeblock: {0}".format(e)) from e
        if not isinstance(obj, dict):
            raise JsonExtractError("Expected JSON object at top-level")
        return obj

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise JsonExtractError("No JSON object delimiters found")

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except Exception as e:
        raise JsonExtractError("Invalid JSON object: {0}".format(e)) from e
    if not isinstance(obj, dict):
        raise JsonExtractError("Expected JSON object at top-level")
    return obj
