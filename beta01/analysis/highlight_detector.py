"""Finds interesting moments in a transcript using an AI model.

The AI only reads transcript text and returns structured suggestions. Every
timestamp and score is validated in Python before it is returned, and the AI
never touches the video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple

from analysis.analysis_errors import (
    CandidateError,
    HighlightDetectionError,
    LLMAuthError,
    LLMError,
    ResponseParseError,
)
from analysis.candidate_validator import validate_candidate
from analysis.clip_models import ClipCandidate
from analysis.highlight_prompts import PARSE_RETRY_REMINDER, build_system_prompt, build_user_prompt
from analysis.json_response_parser import extract_clip_dicts
from analysis.llm_client import LLMClient
from analysis.transcript_chunker import TranscriptChunk, chunk_transcript
from config import AppConfig
from input.transcript_models import Transcript

logger = logging.getLogger(__name__)

PARSE_ATTEMPTS = 2

# Called with (parts finished, total parts, status text).
ProgressCallback = Callable[[int, int, str], None]


class ChatClient(Protocol):
    """Anything with a chat(system_prompt, user_prompt) method."""

    def chat(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class DetectionResult:
    """Outcome of highlight detection."""

    candidates: List[ClipCandidate]
    chunks_total: int
    chunks_failed: int
    rejected_count: int
    warnings: List[str] = field(default_factory=list)


def dedupe_near_identical(
    candidates: List[ClipCandidate], tolerance_seconds: float = 3.0
) -> List[ClipCandidate]:
    """Drop candidates whose start and end are both within the tolerance of a better one.

    Overlapping parts of the transcript can produce the same moment twice. The
    candidate with the higher interestingness is kept. Result is sorted by start.
    Broader overlap handling happens later in the overlap optimizer.
    """
    ranked = sorted(candidates, key=lambda c: c.scores.get("interestingness", 0.0), reverse=True)
    kept: List[ClipCandidate] = []
    for candidate in ranked:
        duplicate = any(
            abs(candidate.start - other.start) <= tolerance_seconds
            and abs(candidate.end - other.end) <= tolerance_seconds
            for other in kept
        )
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.start)


def _analyze_chunk(
    client: ChatClient,
    chunk: TranscriptChunk,
    total_chunks: int,
    config: AppConfig,
    source_name: str,
) -> Tuple[List[ClipCandidate], int]:
    """Analyze one transcript part. Returns (candidates, rejected count)."""
    max_candidates = config.ai.candidates_per_chunk
    system_prompt = build_system_prompt(config.durations, max_candidates)
    user_prompt = build_user_prompt(chunk, total_chunks, source_name, max_candidates)

    raw_items = None
    last_error: Optional[Exception] = None
    for attempt in range(PARSE_ATTEMPTS):
        prompt = user_prompt if attempt == 0 else f"{user_prompt}\n\n{PARSE_RETRY_REMINDER}"
        reply = client.chat(system_prompt, prompt)
        try:
            raw_items = extract_clip_dicts(reply)
            break
        except ResponseParseError as exc:
            last_error = exc
            logger.warning("Part %d: could not read the model's reply (%s).", chunk.index + 1, exc)
    if raw_items is None:
        raise HighlightDetectionError(f"the model's reply could not be parsed ({last_error})")

    candidates: List[ClipCandidate] = []
    rejected = 0
    for raw in raw_items:
        try:
            candidates.append(
                validate_candidate(raw, chunk.start, chunk.end, config.durations, chunk.index)
            )
        except CandidateError as exc:
            rejected += 1
            logger.debug("Part %d: rejected a suggestion: %s", chunk.index + 1, exc)

    candidates.sort(key=lambda c: c.scores["interestingness"], reverse=True)
    return candidates[:max_candidates], rejected


def detect_highlights(
    transcript: Transcript,
    config: AppConfig,
    source_name: str = "",
    client: Optional[ChatClient] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> DetectionResult:
    """Analyze the whole transcript and return validated clip candidates.

    A part that fails is skipped and reported; detection only fails outright when
    every part fails or the provider rejects the credentials.

    Raises:
        HighlightDetectionError: if no part could be analyzed.
    """
    chunks = chunk_transcript(transcript, config.ai.chunk_max_chars, config.ai.chunk_overlap_chars)
    if not chunks:
        raise HighlightDetectionError("The transcript has no segments to analyze.")

    active_client: ChatClient = client if client is not None else LLMClient(config.ai)
    total = len(chunks)
    logger.info(
        "Analyzing the transcript in %d part(s) with model '%s'.", total, config.ai.resolved_model()
    )

    all_candidates: List[ClipCandidate] = []
    warnings: List[str] = []
    failed = 0
    rejected_total = 0

    for chunk in chunks:
        number = chunk.index + 1
        logger.info("Analyzing part %d of %d...", number, total)
        if progress_callback:
            progress_callback(chunk.index, total, f"Analyzing part {number} of {total}")
        try:
            candidates, rejected = _analyze_chunk(active_client, chunk, total, config, source_name)
        except LLMAuthError as exc:
            raise HighlightDetectionError(str(exc)) from exc
        except (LLMError, HighlightDetectionError) as exc:
            failed += 1
            warnings.append(f"Part {number} of {total} failed: {exc}")
            logger.error("Part %d of %d failed: %s", number, total, exc)
            continue
        all_candidates.extend(candidates)
        rejected_total += rejected

    if progress_callback:
        progress_callback(total, total, "Analysis finished")

    if failed == total:
        raise HighlightDetectionError(f"All {total} transcript part(s) failed. {warnings[-1]}")

    unique = dedupe_near_identical(all_candidates)
    if not unique:
        warnings.append("The AI did not suggest any usable highlights.")
    logger.info(
        "Detection finished: %d candidate(s), %d part(s) failed, %d suggestion(s) rejected.",
        len(unique),
        failed,
        rejected_total,
    )
    return DetectionResult(
        candidates=unique,
        chunks_total=total,
        chunks_failed=failed,
        rejected_count=rejected_total,
        warnings=warnings,
    )