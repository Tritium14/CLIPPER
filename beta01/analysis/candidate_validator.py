"""Validates one AI-suggested clip before it enters the rest of the pipeline.

Rules:
    - start and end must parse and end must be after start
    - the clip must lie inside the transcript range (small overshoot is clamped)
    - duration must be at least MIN_PLAUSIBLE_SECONDS and at most
      max_candidate_seconds (default 3x the absolute maximum; the boundary
      optimizer later shortens clips that exceed the real limit)
    - interestingness is required; any other missing score is set to the neutral
      score (50) and recorded in missing_components
    - scores are clamped to 0-100
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from analysis.analysis_errors import CandidateError
from analysis.clip_models import SCORE_COMPONENTS, ClipCandidate
from config import DurationConfig
from input.input_errors import TimestampError
from input.timestamp_utils import parse_timestamp

MIN_PLAUSIBLE_SECONDS = 5.0
NEUTRAL_SCORE = 50.0
MAX_TITLE_CHARS = 120
MAX_REASON_CHARS = 500

_START_KEYS = ("start", "start_time", "start_timestamp", "begin")
_END_KEYS = ("end", "end_time", "end_timestamp", "stop")
_TITLE_KEYS = ("title", "name", "headline")
_REASON_KEYS = ("reason", "why", "rationale", "description", "summary")
_CATEGORY_KEYS = ("category", "type")
_SCORE_KEYS = {
    "interestingness": ("interestingness", "interestingness_score", "interest"),
    "hook_strength": ("hook_strength", "hook", "hook_score"),
    "context_completeness": ("context_completeness", "context", "completeness"),
    "emotional_impact": ("emotional_impact", "emotion", "emotional"),
    "standalone_quality": ("standalone_quality", "standalone"),
    "relevance": ("relevance", "relevance_to_topic"),
}


def _normalize_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key).strip().lower().replace(" ", "_"): value for key, value in raw.items()}


def _first(data: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return None


def parse_time_value(value: Any) -> float:
    """Convert an AI timestamp (seconds as a number, or a timestamp string) to seconds.

    Raises:
        CandidateError: if the value cannot be understood.
    """
    if isinstance(value, bool) or value is None:
        raise CandidateError(f"Invalid time value: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            seconds = float(text)
        except ValueError:
            try:
                seconds = parse_timestamp(text)
            except TimestampError as exc:
                raise CandidateError(f"Invalid time value: {value!r}") from exc
    else:
        raise CandidateError(f"Invalid time value: {value!r}")
    if not math.isfinite(seconds) or seconds < 0:
        raise CandidateError(f"Invalid time value: {value!r}")
    return seconds


def _parse_score(value: Any) -> Optional[float]:
    """Convert a score to 0-100, or None if it cannot be understood."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().rstrip("%").strip()
        try:
            if "/" in text:
                numerator, denominator = text.split("/", 1)
                number = float(numerator) / float(denominator) * 100.0
            else:
                number = float(text)
        except (ValueError, ZeroDivisionError):
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return min(max(number, 0.0), 100.0)


def validate_candidate(
    raw: Dict[str, Any],
    range_start: float,
    range_end: float,
    durations: DurationConfig,
    chunk_index: int = 0,
    tolerance_seconds: float = 5.0,
    max_candidate_seconds: Optional[float] = None,
) -> ClipCandidate:
    """Validate a raw AI clip dict and return a ClipCandidate.

    Args:
        raw: The clip dict from the AI reply.
        range_start, range_end: The time range the AI was shown (the chunk).
        durations: Duration limits from the configuration.
        chunk_index: Which transcript part produced the clip.
        tolerance_seconds: Overshoot past the range that is clamped, not rejected.
        max_candidate_seconds: Longest clip accepted (default 3x absolute maximum).

    Raises:
        CandidateError: if the clip is unusable.
    """
    data = _normalize_keys(raw)
    start = parse_time_value(_first(data, _START_KEYS))
    end = parse_time_value(_first(data, _END_KEYS))
    if end <= start:
        raise CandidateError(f"End ({end:.1f}s) is not after start ({start:.1f}s).")

    if start < range_start - tolerance_seconds or end > range_end + tolerance_seconds:
        raise CandidateError(
            f"Clip {start:.1f}-{end:.1f}s lies outside the transcript range "
            f"{range_start:.1f}-{range_end:.1f}s."
        )
    start = max(start, range_start)
    end = min(end, range_end)

    duration = end - start
    if duration < MIN_PLAUSIBLE_SECONDS:
        raise CandidateError(f"Clip is only {duration:.1f}s long.")
    limit = max_candidate_seconds if max_candidate_seconds is not None else 3 * durations.absolute_max_seconds
    if duration > limit:
        raise CandidateError(f"Clip is {duration:.0f}s long, more than the {limit:.0f}s limit.")

    scores: Dict[str, float] = {}
    missing = []
    for component in SCORE_COMPONENTS:
        score = _parse_score(_first(data, _SCORE_KEYS[component]))
        if score is None:
            if component == "interestingness":
                raise CandidateError("The clip has no interestingness score.")
            score = NEUTRAL_SCORE
            missing.append(component)
        scores[component] = score

    title = " ".join(str(_first(data, _TITLE_KEYS) or "").split())[:MAX_TITLE_CHARS] or "Untitled clip"
    reason = " ".join(str(_first(data, _REASON_KEYS) or "").split())[:MAX_REASON_CHARS]
    category = " ".join(str(_first(data, _CATEGORY_KEYS) or "").split()).lower()[:40]

    return ClipCandidate(
        start=start,
        end=end,
        title=title,
        reason=reason,
        scores=scores,
        category=category,
        chunk_index=chunk_index,
        missing_components=tuple(missing),
    )