"""Extracts clip suggestions from an AI reply, tolerating common formatting problems.

Handles markdown code fences, <think> blocks, a wrapper object such as
{"clips": [...]}, trailing commas, text around the JSON, and replies cut off by
the token limit (complete objects are salvaged).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from analysis.analysis_errors import ResponseParseError

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_START_KEYS = ("start", "start_time", "start_timestamp", "begin")
_WRAPPER_KEYS = ("clips", "highlights", "candidates", "moments", "segments", "results")
_NOT_PARSED = object()


def strip_wrappers(text: str) -> str:
    """Remove <think> blocks and markdown fences from a reply."""
    text = _THINK_RE.sub("", text or "")
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    return text.strip()


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", text))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


def _looks_like_clip(obj: Dict[str, Any]) -> bool:
    keys = {str(k).strip().lower() for k in obj}
    return any(key in keys for key in _START_KEYS)


def _scan_object(text: str, start: int) -> Optional[int]:
    """Return the index of the '}' that closes the object opening at start, or None."""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def salvage_objects(text: str) -> List[Dict[str, Any]]:
    """Find every complete JSON object in the text that looks like a clip."""
    found: List[Dict[str, Any]] = []
    position = 0
    while True:
        position = text.find("{", position)
        if position == -1:
            break
        end = _scan_object(text, position)
        if end is None:
            position += 1
            continue
        try:
            obj = _loads(text[position : end + 1])
        except ValueError:
            position += 1
            continue
        if isinstance(obj, dict) and _looks_like_clip(obj):
            found.append(obj)
            position = end + 1
        else:
            position += 1
    return found


def _items_from_json(data: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull the list of clip dicts out of parsed JSON, or None if there is none."""
    if isinstance(data, list):
        if not data:
            return []
        items = [item for item in data if isinstance(item, dict)]
        return items or None
    if isinstance(data, dict):
        for key in _WRAPPER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if _looks_like_clip(data):
            return [data]
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
                return value
    return None


def extract_clip_dicts(reply: str) -> List[Dict[str, Any]]:
    """Return the clip dicts found in an AI reply (an empty list is a valid answer).

    Raises:
        ResponseParseError: if the reply is empty or contains no usable JSON.
    """
    cleaned = strip_wrappers(reply)
    if not cleaned:
        raise ResponseParseError("The model returned an empty reply.")

    try:
        data: Any = _loads(cleaned)
    except ValueError:
        data = _NOT_PARSED

    if data is not _NOT_PARSED:
        items = _items_from_json(data)
        if items is not None:
            return items

    salvaged = salvage_objects(cleaned)
    if salvaged:
        return salvaged

    preview = cleaned[:120].replace("\n", " ")
    raise ResponseParseError(f"No usable JSON found in the model's reply: '{preview}'")