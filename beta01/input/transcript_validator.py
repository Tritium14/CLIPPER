"""Cleans and validates transcript segments.

Steps, in order: remove invalid starts, collapse rolling auto-captions, flatten
text and drop empty segments, sort, merge duplicates, repair or estimate end
times, and trim overlaps. Every change is reported as a warning string.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Tuple

from input.timestamp_utils import format_timestamp
from input.transcript_models import TranscriptSegment

logger = logging.getLogger(__name__)

# An echo cue is a tiny cue that repeats the previous cue's last line.
_ECHO_MAX_SECONDS = 0.1


def collapse_rolling_captions(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Remove repeated lines from rolling auto-captions (such as YouTube VTT).

    In rolling captions each cue repeats the previous cue's last line before
    adding a new one. Only multi-line cues are collapsed, plus tiny single-line
    echo cues, so genuine repeated speech is kept.
    """
    result: List[TranscriptSegment] = []
    previous_lines: List[str] = []
    for segment in segments:
        lines = [line.strip() for line in segment.text.split("\n") if line.strip()]
        new_lines = lines
        if len(lines) >= 2:
            for k in range(min(len(lines), len(previous_lines)), 0, -1):
                if lines[:k] == previous_lines[-k:]:
                    new_lines = lines[k:]
                    break
        elif (
            len(lines) == 1
            and previous_lines
            and lines == previous_lines[-1:]
            and segment.end is not None
            and segment.end - segment.start <= _ECHO_MAX_SECONDS
        ):
            new_lines = []
        previous_lines = lines
        result.append(replace(segment, text="\n".join(new_lines)))
    return result


def normalize_segments(
    segments: List[TranscriptSegment],
    merge_gap_seconds: float = 0.05,
    words_per_second: float = 2.5,
    min_estimated_seconds: float = 1.0,
    max_estimated_seconds: float = 8.0,
) -> Tuple[List[TranscriptSegment], List[str]]:
    """Clean raw segments. Returns (segments, warnings).

    The returned segments are sorted, non-empty, free of duplicates, and all
    have an end time greater than their start time.
    """
    notes: List[str] = []

    valid = [s for s in segments if s.start >= 0]
    if len(valid) != len(segments):
        notes.append(f"Dropped {len(segments) - len(valid)} segment(s) with negative start times.")

    collapsed = collapse_rolling_captions(valid)

    flattened: List[TranscriptSegment] = []
    empty = 0
    for segment in collapsed:
        text = " ".join(segment.text.split())
        if not text:
            empty += 1
            continue
        flattened.append(replace(segment, text=text))
    if empty:
        notes.append(f"Dropped {empty} empty or repeated caption segment(s).")

    ordered = sorted(flattened, key=lambda s: s.start)
    if ordered != flattened:
        notes.append("Segments were out of order and have been sorted by start time.")

    deduped: List[TranscriptSegment] = []
    duplicates = 0
    for segment in ordered:
        if deduped and deduped[-1].text == segment.text:
            previous = deduped[-1]
            if segment.start <= previous.effective_end + merge_gap_seconds:
                ends = [e for e in (previous.end, segment.end) if e is not None]
                deduped[-1] = replace(previous, end=max(ends) if ends else None)
                duplicates += 1
                continue
        deduped.append(segment)
    if duplicates:
        notes.append(f"Merged {duplicates} duplicate segment(s).")

    repaired: List[TranscriptSegment] = []
    invalid_ends = 0
    for segment in deduped:
        if segment.end is not None and segment.end <= segment.start:
            repaired.append(replace(segment, end=None))
            invalid_ends += 1
        else:
            repaired.append(segment)
    if invalid_ends:
        notes.append(f"{invalid_ends} segment(s) had an end time at or before their start; ends were estimated.")

    filled: List[TranscriptSegment] = []
    estimated = 0
    for index, segment in enumerate(repaired):
        if segment.end is None:
            next_start = repaired[index + 1].start if index + 1 < len(repaired) else None
            if next_start is not None and next_start > segment.start:
                end = next_start
            else:
                words = len(segment.text.split())
                end = segment.start + min(
                    max(words / words_per_second, min_estimated_seconds), max_estimated_seconds
                )
            filled.append(replace(segment, end=end, end_estimated=True))
            estimated += 1
        else:
            filled.append(segment)
    if estimated:
        notes.append(f"Estimated end times for {estimated} segment(s) that had none.")

    clamped = 0
    for index in range(len(filled) - 1):
        current, following = filled[index], filled[index + 1]
        if current.end is not None and current.end > following.start > current.start:
            filled[index] = replace(current, end=following.start)
            clamped += 1
    if clamped:
        notes.append(f"Trimmed {clamped} overlapping segment(s) so they end when the next one starts.")

    for note in notes:
        logger.debug("Transcript normalization: %s", note)
    return filled, notes


def fit_to_video_duration(
    segments: List[TranscriptSegment], video_duration: float
) -> Tuple[List[TranscriptSegment], List[str]]:
    """Drop or trim segments beyond the end of the video and flag mismatches."""
    if video_duration <= 0:
        return segments, []

    notes: List[str] = []
    kept: List[TranscriptSegment] = []
    dropped = 0
    for segment in segments:
        if segment.start >= video_duration:
            dropped += 1
            continue
        if segment.effective_end > video_duration:
            segment = replace(segment, end=video_duration)
        kept.append(segment)

    if dropped:
        notes.append(
            f"Dropped {dropped} segment(s) starting after the end of the video "
            f"({format_timestamp(video_duration)}). The transcript may belong to a different video."
        )
    if kept:
        last_end = max(s.effective_end for s in kept)
        if last_end < 0.5 * video_duration:
            notes.append(
                f"The transcript ends at {format_timestamp(last_end)} but the video is "
                f"{format_timestamp(video_duration)} long. The transcript may be incomplete."
            )
    return kept, notes