"""Parses SRT, VTT, and timestamped plain-text transcripts.

Public interface:
    load_transcript(path, video_duration=None) -> Transcript
    parse_transcript_text(text, source_format=None, video_duration=None) -> Transcript
    detect_format(text) -> "srt" | "vtt" | "text"

Plain-text lines may look like:
    [00:01:23] text         00:01:23 text         0:01:23 - text
    [00:01:23 - 00:01:30] text         00:01:23 --> 00:01:30 text
    0:01:23                 (timestamp alone on a line, text on the lines below)
Two-part timestamps are read as minutes:seconds.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from input.input_errors import TimestampError, TranscriptError
from input.timestamp_utils import TIMESTAMP_PATTERN, parse_timestamp
from input.transcript_models import Transcript, TranscriptSegment
from input.transcript_validator import fit_to_video_duration, normalize_segments

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("srt", "vtt", "text")

_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>|<\d{1,3}:\d{2}(?::\d{2})?[.,]\d{1,3}>")
_STYLE_RE = re.compile(r"\{\\[^}]*\}")
_CUE_TIMING_RE = re.compile(r"^\s*(?P<start>\S+)\s*-->\s*(?P<end>\S+)(?P<rest>.*)$")
_CUE_LINE_ONLY_RE = re.compile(
    r"^\s*" + TIMESTAMP_PATTERN + r"\s*-->\s*" + TIMESTAMP_PATTERN + r"(?:\s+[A-Za-z]+:\S+)*\s*$"
)
_BLOCK_START_RE = re.compile(r"^(NOTE|STYLE|REGION)\b")
_TEXT_LINE_RE = re.compile(
    r"^[\[\(\{]?\s*(?P<start>" + TIMESTAMP_PATTERN + r")"
    r"(?:\s*(?:-->|-|–|—|\bto\b)\s*(?P<end>" + TIMESTAMP_PATTERN + r"))?"
    r"\s*[\]\)\}]?\s*[-–—:|]?\s*(?P<text>.*)$"
)

_FORMAT_HINT = (
    "Supported formats: SRT, VTT, or plain text where each line starts with a "
    "timestamp such as [00:01:23] or 0:01:23."
)


def _normalize_newlines(text: str) -> str:
    return text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")


def _clean_line(line: str, strip_markup: bool) -> str:
    if strip_markup:
        line = _TAG_RE.sub("", line)
        line = _STYLE_RE.sub("", line)
    line = html.unescape(line)
    return " ".join(line.split())


def detect_format(text: str) -> str:
    """Guess the transcript format from its content."""
    cleaned = _normalize_newlines(text)
    lines = cleaned.split("\n")
    for line in lines:
        if line.strip():
            if line.strip().startswith("WEBVTT"):
                return "vtt"
            break
    for line in lines:
        if _CUE_LINE_ONLY_RE.match(line):
            return "srt" if re.search(r"\d,\d{1,3}\s*-->", line) else "vtt"
    return "text"


class _Cue:
    """A cue being assembled while parsing."""

    def __init__(self, start: float, end: Optional[float]) -> None:
        self.start = start
        self.end = end
        self.lines: List[str] = []

    def to_segment(self) -> TranscriptSegment:
        return TranscriptSegment(self.start, self.end, "\n".join(self.lines))


def _parse_cue_blocks(text: str, is_vtt: bool) -> Tuple[List[TranscriptSegment], int]:
    """Parse SRT or VTT. Returns (raw segments, number of malformed cues skipped)."""
    lines = text.split("\n")
    segments: List[TranscriptSegment] = []
    skipped = 0
    current: Optional[_Cue] = None
    discarding = False

    def finish() -> None:
        nonlocal current
        if current is not None:
            segments.append(current.to_segment())
        current = None

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            finish()
            discarding = False
            continue
        if discarding:
            continue

        if "-->" in line:
            match = _CUE_TIMING_RE.match(line)
            try:
                if not match:
                    raise TimestampError("Malformed cue timing line.")
                start = parse_timestamp(match.group("start"))
                end = parse_timestamp(match.group("end"))
            except TimestampError:
                finish()
                skipped += 1
                discarding = True
                continue
            finish()
            current = _Cue(start, end)
            continue

        if current is None:
            if is_vtt and _BLOCK_START_RE.match(line):
                discarding = True
            continue  # cue index, cue identifier, or header metadata

        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if line.isdigit() and "-->" in next_line:
            finish()  # an SRT index that appears without a blank line before it
            continue

        cleaned = _clean_line(line, strip_markup=True)
        if cleaned:
            current.lines.append(cleaned)

    finish()
    return segments, skipped


def _parse_plain_text(text: str) -> Tuple[List[TranscriptSegment], int]:
    """Parse timestamped plain text. Returns (raw segments, lines skipped)."""
    segments: List[TranscriptSegment] = []
    skipped = 0
    current: Optional[_Cue] = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            segments.append(current.to_segment())
        current = None

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        match = _TEXT_LINE_RE.match(line)
        if match:
            try:
                start = parse_timestamp(match.group("start"))
                end = parse_timestamp(match.group("end")) if match.group("end") else None
            except TimestampError:
                finish()
                skipped += 1
                continue
            finish()
            current = _Cue(start, end)
            rest = _clean_line(match.group("text"), strip_markup=False)
            if rest:
                current.lines.append(rest)
            continue

        if current is None:
            skipped += 1  # text before the first timestamp
            continue
        cleaned = _clean_line(line, strip_markup=False)
        if cleaned:
            current.lines.append(cleaned)

    finish()
    return segments, skipped


def parse_transcript_text(
    text: str,
    source_format: Optional[str] = None,
    video_duration: Optional[float] = None,
) -> Transcript:
    """Parse and validate transcript text.

    Args:
        text: The transcript contents.
        source_format: "srt", "vtt", or "text". Detected automatically if None.
        video_duration: Video length in seconds. When given, segments beyond the
            end of the video are dropped and mismatches are reported.

    Raises:
        TranscriptError: if the transcript is empty, has an unsupported format,
            or contains no usable timestamped text.
    """
    if not text or not text.strip():
        raise TranscriptError("The transcript is empty.")

    cleaned = _normalize_newlines(text)
    fmt = (source_format or "").strip().lower() or detect_format(cleaned)
    if fmt not in SUPPORTED_FORMATS:
        raise TranscriptError(f"Unsupported transcript format '{fmt}'. {_FORMAT_HINT}")

    if fmt == "text":
        raw_segments, skipped = _parse_plain_text(cleaned)
    else:
        raw_segments, skipped = _parse_cue_blocks(cleaned, is_vtt=(fmt == "vtt"))

    warnings: List[str] = []
    if skipped:
        warnings.append(f"Skipped {skipped} line(s) or cue(s) with missing or malformed timestamps.")
    if not raw_segments:
        raise TranscriptError(f"No timestamped segments were found (read as {fmt}). {_FORMAT_HINT}")

    segments, notes = normalize_segments(raw_segments)
    warnings.extend(notes)
    if not segments:
        raise TranscriptError("The transcript contained no usable text after cleanup.")

    if video_duration is not None:
        segments, notes = fit_to_video_duration(segments, video_duration)
        warnings.extend(notes)
        if not segments:
            raise TranscriptError(
                "No transcript segments fall inside the video. "
                "The transcript may belong to a different video."
            )

    for warning in warnings:
        logger.warning("Transcript: %s", warning)
    logger.info("Parsed %d transcript segments (%s).", len(segments), fmt)
    return Transcript(segments=segments, source_format=fmt, warnings=warnings)


def _decode_bytes(data: bytes) -> Tuple[str, List[str]]:
    """Decode file bytes, trying UTF-8 first. Returns (text, notes)."""
    notes: List[str] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), notes
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig"), notes
    except UnicodeDecodeError:
        notes.append("The file is not UTF-8, so it was read as Windows-1252 (some characters may look wrong).")
        return data.decode("cp1252", errors="replace"), notes


def load_transcript(path_text: str, video_duration: Optional[float] = None) -> Transcript:
    """Read, parse, and validate a transcript file.

    Raises:
        TranscriptError: if the file is missing, unreadable, empty, or unusable.
    """
    cleaned = (path_text or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        raise TranscriptError("No transcript file path was provided.")

    path = Path(cleaned).expanduser()
    if not path.exists():
        raise TranscriptError(f"Transcript file not found: {path}")
    if not path.is_file():
        raise TranscriptError(f"Not a file: {path}")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TranscriptError(f"Cannot read transcript file '{path}': {exc}") from exc
    if not data.strip():
        raise TranscriptError(f"The transcript file is empty: {path}")

    text, notes = _decode_bytes(data)
    transcript = parse_transcript_text(text, video_duration=video_duration)
    transcript.warnings = notes + transcript.warnings
    transcript.source_path = path.resolve()
    return transcript