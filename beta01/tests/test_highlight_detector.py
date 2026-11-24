"""Tests for analysis/highlight_detector.py (a fake AI client is used)."""

import json

import pytest

from analysis.analysis_errors import HighlightDetectionError, LLMAuthError, LLMError
from analysis.clip_models import SCORE_COMPONENTS, ClipCandidate
from analysis.highlight_detector import dedupe_near_identical, detect_highlights
from analysis.transcript_chunker import chunk_transcript
from config import AppConfig
from input.transcript_models import Transcript, TranscriptSegment


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_transcript():
    segments = [
        TranscriptSegment(i * 10.0, i * 10.0 + 10.0, f"Segment number {i} says something.")
        for i in range(60)
    ]
    return Transcript(segments, "text")


def single_chunk_config():
    config = AppConfig()
    config.ai.chunk_max_chars = 100000
    return config


def multi_chunk_config():
    config = AppConfig()
    config.ai.chunk_max_chars = 700
    config.ai.chunk_overlap_chars = 100
    return config


def clip(start, end, interest=80):
    return {
        "start": start,
        "end": end,
        "title": f"Clip {start}",
        "reason": "r",
        "interestingness": interest,
        "hook_strength": 60,
        "context_completeness": 60,
    }


def reply(*clips):
    return json.dumps(list(clips))


def candidate(start, end, interest):
    scores = {name: 50.0 for name in SCORE_COMPONENTS}
    scores["interestingness"] = interest
    return ClipCandidate(start=start, end=end, title="t", reason="", scores=scores)


def test_single_chunk_success():
    client = FakeClient([reply(clip("0:01:00", "0:02:00"))])
    result = detect_highlights(make_transcript(), single_chunk_config(), "Demo", client)
    assert len(result.candidates) == 1
    assert result.candidates[0].start == 60.0
    assert result.chunks_total == 1
    assert result.chunks_failed == 0
    assert "emotional_impact" in result.candidates[0].missing_components
    assert "Demo" in client.prompts[0]


def test_invalid_suggestions_are_rejected_and_counted():
    bad = clip("0:03:00", "0:02:00")
    client = FakeClient([reply(bad, clip("0:01:00", "0:02:00"))])
    result = detect_highlights(make_transcript(), single_chunk_config(), "", client)
    assert len(result.candidates) == 1
    assert result.rejected_count == 1


def test_unreadable_reply_is_retried_once():
    client = FakeClient(["Sorry, no.", reply(clip("0:01:00", "0:02:00"))])
    result = detect_highlights(make_transcript(), single_chunk_config(), "", client)
    assert len(result.candidates) == 1
    assert len(client.prompts) == 2
    assert "ONLY" in client.prompts[1]


def test_empty_answer_is_not_an_error():
    client = FakeClient(["[]"])
    result = detect_highlights(make_transcript(), single_chunk_config(), "", client)
    assert result.candidates == []
    assert any("did not suggest" in w for w in result.warnings)


def test_failed_part_is_skipped_and_reported():
    config = multi_chunk_config()
    transcript = make_transcript()
    chunks = chunk_transcript(transcript, config.ai.chunk_max_chars, config.ai.chunk_overlap_chars)
    assert len(chunks) >= 3
    replies = [LLMError("boom")] + [
        reply(clip(c.start + 10, c.start + 60)) for c in chunks[1:]
    ]
    result = detect_highlights(transcript, config, "", FakeClient(replies))
    assert result.chunks_failed == 1
    assert len(result.candidates) == len(chunks) - 1
    assert any("Part 1 of" in w for w in result.warnings)


def test_all_parts_failing_raises():
    with pytest.raises(HighlightDetectionError):
        detect_highlights(make_transcript(), single_chunk_config(), "", FakeClient([LLMError("down")]))


def test_unparseable_replies_for_every_part_raise():
    client = FakeClient(["garbage", "garbage"])
    with pytest.raises(HighlightDetectionError):
        detect_highlights(make_transcript(), single_chunk_config(), "", client)


def test_auth_error_aborts_immediately():
    client = FakeClient([LLMAuthError("bad key")])
    with pytest.raises(HighlightDetectionError):
        detect_highlights(make_transcript(), multi_chunk_config(), "", client)
    assert len(client.prompts) == 1


def test_candidates_per_chunk_keeps_the_best():
    config = single_chunk_config()
    config.ai.candidates_per_chunk = 3
    clips = [clip(20 + i * 40, 50 + i * 40, interest=50 + i) for i in range(8)]
    result = detect_highlights(make_transcript(), config, "", FakeClient([reply(*clips)]))
    assert [c.start for c in result.candidates] == [220.0, 260.0, 300.0]


def test_dedupe_keeps_the_higher_scored_duplicate():
    kept = dedupe_near_identical([candidate(60, 120, 70), candidate(61, 121, 90), candidate(300, 360, 50)])
    assert [(c.start, c.scores["interestingness"]) for c in kept] == [(61, 90), (300, 50)]


def test_progress_callback_reports_every_part():
    events = []
    client = FakeClient([reply(clip("0:01:00", "0:02:00"))])
    detect_highlights(
        make_transcript(), single_chunk_config(), "", client, lambda d, t, s: events.append((d, t))
    )
    assert events == [(0, 1), (1, 1)]