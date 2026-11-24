"""Tests for input/transcript_validator.py and transcript_models.py."""

import pytest

from input.transcript_models import Transcript, TranscriptSegment
from input.transcript_validator import fit_to_video_duration, normalize_segments


def seg(start, end, text):
    return TranscriptSegment(start, end, text)


def texts(segments):
    return [s.text for s in segments]


def test_rolling_captions_are_collapsed():
    raw = [
        seg(0.0, 3.0, "line one"),
        seg(2.0, 5.0, "line one\nline two"),
        seg(5.0, 5.01, "line two"),
        seg(5.01, 8.0, "line two\nline three"),
    ]
    result, _ = normalize_segments(raw)
    assert texts(result) == ["line one", "line two", "line three"]


def test_repeated_single_line_speech_is_kept():
    result, _ = normalize_segments([seg(1, 2, "no"), seg(10, 11, "no")])
    assert texts(result) == ["no", "no"]


def test_exact_duplicates_are_merged():
    result, notes = normalize_segments([seg(1, 2, "hi"), seg(1, 2, "hi")])
    assert texts(result) == ["hi"]
    assert any("duplicate" in n for n in notes)


def test_empty_segments_dropped():
    result, notes = normalize_segments([seg(1, 2, "  "), seg(3, 4, "real")])
    assert texts(result) == ["real"]
    assert any("empty" in n for n in notes)


def test_negative_starts_dropped():
    result, _ = normalize_segments([seg(-1, 2, "bad"), seg(3, 4, "ok")])
    assert texts(result) == ["ok"]


def test_out_of_order_segments_are_sorted():
    result, notes = normalize_segments([seg(10, 12, "b"), seg(1, 3, "a")])
    assert texts(result) == ["a", "b"]
    assert any("sorted" in n for n in notes)


def test_missing_ends_are_estimated():
    result, _ = normalize_segments([seg(0, None, "a b"), seg(5, None, "c")])
    assert result[0].end == pytest.approx(5.0)
    assert result[0].end_estimated
    assert result[1].end == pytest.approx(6.0)  # minimum estimate of 1 second
    assert result[1].end_estimated


def test_invalid_end_is_replaced():
    result, notes = normalize_segments([seg(5, 4, "x")])
    assert result[0].end == pytest.approx(6.0)
    assert result[0].end_estimated
    assert any("at or before" in n for n in notes)


def test_overlaps_are_trimmed():
    result, _ = normalize_segments([seg(0, 10, "a"), seg(5, 8, "b")])
    assert result[0].end == pytest.approx(5.0)
    assert result[1].end == pytest.approx(8.0)


def test_fit_to_video_trims_and_drops():
    segments = [seg(0, 10, "a"), seg(95, 120, "b"), seg(200, 210, "c")]
    kept, notes = fit_to_video_duration(segments, 100.0)
    assert texts(kept) == ["a", "b"]
    assert kept[1].end == pytest.approx(100.0)
    assert any("different video" in n for n in notes)


def test_fit_to_video_flags_incomplete_transcript():
    kept, notes = fit_to_video_duration([seg(0, 10, "a")], 100.0)
    assert len(kept) == 1
    assert any("incomplete" in n for n in notes)


def test_fit_to_video_ignores_zero_duration():
    segments = [seg(0, 10, "a")]
    kept, notes = fit_to_video_duration(segments, 0.0)
    assert kept == segments and notes == []


def test_transcript_helpers():
    transcript = Transcript(
        segments=[seg(0, 5, "hello world"), seg(5, 10, "second part"), seg(20, 25, "late")],
        source_format="srt",
    )
    assert transcript.word_count == 5
    assert transcript.start_seconds == 0.0
    assert transcript.end_seconds == 25.0
    assert texts(transcript.segments_in_range(4, 6)) == ["hello world", "second part"]
    assert transcript.text_in_range(10, 20) == ""
    assert transcript.text_in_range(19, 21) == "late"
    assert "3 segments" in transcript.summary()