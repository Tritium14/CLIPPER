"""Internal representation of a transcript, shared by all later stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from input.timestamp_utils import format_timestamp


@dataclass(frozen=True)
class TranscriptSegment:
    """One timed piece of transcript text.

    After normalization every segment has an end time. end_estimated is True when
    the end time was not in the source file and was inferred.
    """

    start: float
    end: Optional[float]
    text: str
    end_estimated: bool = False

    @property
    def effective_end(self) -> float:
        return self.end if self.end is not None else self.start

    @property
    def duration(self) -> float:
        return max(self.effective_end - self.start, 0.0)


@dataclass
class Transcript:
    """A validated, normalized transcript."""

    segments: List[TranscriptSegment]
    source_format: str
    warnings: List[str] = field(default_factory=list)
    source_path: Optional[Path] = None

    @property
    def start_seconds(self) -> float:
        return self.segments[0].start if self.segments else 0.0

    @property
    def end_seconds(self) -> float:
        return max((s.effective_end for s in self.segments), default=0.0)

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def segments_in_range(self, start: float, end: float) -> List[TranscriptSegment]:
        """Segments that overlap the time range [start, end)."""
        return [s for s in self.segments if s.effective_end > start and s.start < end]

    def text_in_range(self, start: float, end: float) -> str:
        """All transcript text overlapping the time range, joined with spaces."""
        return " ".join(s.text for s in self.segments_in_range(start, end))

    def preview(self, count: int = 5) -> List[str]:
        """The first few segments formatted for display."""
        lines = []
        for segment in self.segments[:count]:
            span = f"{format_timestamp(segment.start)}-{format_timestamp(segment.effective_end)}"
            text = segment.text if len(segment.text) <= 80 else segment.text[:77] + "..."
            lines.append(f"  [{span}] {text}")
        return lines

    def summary(self) -> str:
        """Multi-line summary for display."""
        lines = [
            f"Transcript: {len(self.segments)} segments | format {self.source_format} | "
            f"{format_timestamp(self.start_seconds)} to {format_timestamp(self.end_seconds)} | "
            f"{self.word_count} words"
        ]
        for warning in self.warnings:
            lines.append(f"  Warning: {warning}")
        lines.append("First segments:")
        lines.extend(self.preview())
        return "\n".join(lines)