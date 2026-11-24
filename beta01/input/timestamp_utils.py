"""Parsing and formatting of timestamps used in transcripts."""

from __future__ import annotations

import re

from input.input_errors import TimestampError

# One timestamp such as 12:34, 1:02:03, 00:01:23,456, or 01:23.5
TIMESTAMP_PATTERN = r"\d{1,3}(?::\d{1,3}){1,2}(?:[.,]\d{1,3})?"

_COLON_RE = re.compile(r"^(?:(\d{1,3}):)?(\d{1,3}):(\d{1,2})(?:[.,](\d{1,3}))?$")
_UNIT_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$", re.IGNORECASE)


def parse_timestamp(text: str) -> float:
    """Convert a timestamp string to seconds.

    Supported forms:
        HH:MM:SS,mmm   HH:MM:SS.mmm   H:MM:SS
        MM:SS.mmm      MM:SS          (two parts always mean minutes:seconds)
        90s   1h2m3s   123.5s
    Surrounding [] or () are ignored.

    Raises:
        TimestampError: if the text is not a valid timestamp.
    """
    cleaned = (text or "").strip().strip("[]()").strip()
    if not cleaned:
        raise TimestampError("Empty timestamp.")

    match = _COLON_RE.match(cleaned)
    if match:
        hours_text, minutes_text, seconds_text, fraction_text = match.groups()
        hours = int(hours_text) if hours_text else 0
        minutes = int(minutes_text)
        seconds = int(seconds_text)
        if seconds >= 60:
            raise TimestampError(f"Seconds out of range in timestamp '{text}'.")
        if hours_text is not None and minutes >= 60:
            raise TimestampError(f"Minutes out of range in timestamp '{text}'.")
        millis = int(fraction_text.ljust(3, "0")) if fraction_text else 0
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    unit = _UNIT_RE.match(cleaned)
    if unit and any(unit.groups()):
        hours_text, minutes_text, seconds_text = unit.groups()
        return int(hours_text or 0) * 3600 + int(minutes_text or 0) * 60 + float(seconds_text or 0)

    raise TimestampError(f"Unrecognized timestamp: '{text}'.")


def format_timestamp(seconds: float, millis: bool = False) -> str:
    """Format seconds as H:MM:SS (or H:MM:SS.mmm when millis is True)."""
    seconds = max(float(seconds), 0.0)
    if millis:
        total_ms = int(round(seconds * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"