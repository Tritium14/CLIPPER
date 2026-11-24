"""Prompt text for highlight detection, kept separate so it is easy to tune."""

from __future__ import annotations

from analysis.transcript_chunker import TranscriptChunk
from config import DurationConfig
from input.timestamp_utils import format_timestamp

PARSE_RETRY_REMINDER = (
    "Your previous reply could not be read. Reply with ONLY a valid JSON array and nothing else."
)

_SYSTEM_TEMPLATE = """You are an expert editor of short-form vertical video (YouTube Shorts, Instagram Reels, TikTok). You read a timestamped transcript of a long video and choose the moments that would make the best standalone short clips.

Judge meaning and context, not keywords. Look for:
- strong hooks that grab attention in the first seconds
- surprising revelations and unexpected conclusions
- interesting stories with a clear payoff
- funny or emotional moments
- controversial opinions
- useful, concrete advice
- memorable quotes and engaging explanations

Avoid: boring introductions, repetitive statements, incomplete thoughts, quotes that need missing context, filler, and anything that would not make sense without the rest of the video.

Rules for every clip:
- Start where the idea begins (include any setup needed to understand it) and end after the point or payoff is complete. Start and end at natural sentence boundaries.
- Length: aim for <<MIN>> to <<PREF>> seconds. Never longer than <<ABS>> seconds.
- Clips must not overlap each other.
- Copy timestamps from the transcript markers and write them as H:MM:SS (for example 0:12:34).
- Choose at most <<COUNT>> clips. Return fewer, or an empty list, if few moments are truly good. Do not pad the list.

Each score is an integer from 0 to 100:
- interestingness: how engaging the clip is overall
- hook_strength: how well the first seconds grab attention
- context_completeness: how complete and self-contained the thought is
- emotional_impact: how strongly it affects the viewer
- standalone_quality: how well it makes sense without the rest of the video
- relevance: how central it is to the video's main subject

category must be one of: story, funny, emotional, controversial, advice, quote, revelation, explanation, other.

Reply with ONLY a JSON array. No markdown, no explanations. Each element must look like this:
{"start": "0:12:34", "end": "0:13:40", "title": "Short descriptive title", "reason": "One or two sentences on why this works as a short", "category": "story", "interestingness": 85, "hook_strength": 80, "context_completeness": 90, "emotional_impact": 60, "standalone_quality": 85, "relevance": 75}"""


def build_system_prompt(durations: DurationConfig, max_candidates: int) -> str:
    """Build the instructions that define the task and the reply format."""
    return (
        _SYSTEM_TEMPLATE.replace("<<MIN>>", f"{durations.min_seconds:.0f}")
        .replace("<<PREF>>", f"{durations.preferred_max_seconds:.0f}")
        .replace("<<ABS>>", f"{durations.absolute_max_seconds:.0f}")
        .replace("<<COUNT>>", str(max_candidates))
    )


def build_user_prompt(
    chunk: TranscriptChunk, total_chunks: int, source_name: str, max_candidates: int
) -> str:
    """Build the message that carries one transcript part."""
    name = source_name or "Untitled video"
    return (
        f"Video: {name}\n"
        f"This is part {chunk.index + 1} of {total_chunks} of the transcript, covering "
        f"{format_timestamp(chunk.start)} to {format_timestamp(chunk.end)}. Parts overlap slightly, "
        f"so skip moments that are clearly cut off at the edges of this part.\n"
        f"Pick up to {max_candidates} of the best moments.\n\n"
        f"TRANSCRIPT START\n{chunk.text}\nTRANSCRIPT END\n\n"
        f"Reply with ONLY the JSON array."
    )