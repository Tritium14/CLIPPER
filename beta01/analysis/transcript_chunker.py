"""Splits a transcript into overlapping parts sized for the AI model's context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from input.timestamp_utils import format_timestamp
from input.transcript_models import Transcript, TranscriptSegment


@dataclass(frozen=True)
class TranscriptChunk:
    """One part of the transcript, formatted for the AI prompt."""

    index: int
    start: float
    end: float
    text: str
    segment_count: int


def format_segment_line(segment: TranscriptSegment) -> str:
    """Format a segment as '[H:MM:SS] text' for the prompt."""
    return f"[{format_timestamp(segment.start)}] {segment.text}"


def chunk_transcript(transcript: Transcript, max_chars: int, overlap_chars: int) -> List[TranscriptChunk]:
    """Split the transcript into chunks of whole segments.

    Each chunk holds at most max_chars characters (a single oversized segment
    still gets its own chunk). Consecutive chunks share roughly overlap_chars of
    text so that moments near a boundary are seen with context on both sides.
    """
    segments = transcript.segments
    if not segments:
        return []

    lines = [format_segment_line(s) for s in segments]
    lengths = [len(line) + 1 for line in lines]
    count = len(segments)
    chunks: List[TranscriptChunk] = []
    i = 0
    while i < count:
        j = i
        total = 0
        while j < count and (j == i or total + lengths[j] <= max_chars):
            total += lengths[j]
            j += 1
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                start=segments[i].start,
                end=segments[j - 1].effective_end,
                text="\n".join(lines[i:j]),
                segment_count=j - i,
            )
        )
        if j >= count:
            break
        k = j
        accumulated = 0
        while k - 1 > i and accumulated + lengths[k - 1] <= overlap_chars:
            k -= 1
            accumulated += lengths[k]
        i = max(k, i + 1)
    return chunks