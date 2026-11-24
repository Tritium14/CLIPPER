"""Tests for analysis/candidate_validator.py."""

import pytest

from analysis.analysis_errors import CandidateError
from analysis.candidate_validator import NEUTRAL_SCORE, parse_time_value, validate_candidate
from config import DurationConfig

DURATIONS = DurationConfig()


def raw(**overrides):
    base = {
        "start": "0:01:00",
        "end": "0:02:00",
        "title": "A title",
        "reason": "A reason",
        "interestingness": 80,
        "hook_strength": 70,
        "context_completeness": 90,
        "emotional_impact": 60,
        "standalone_quality": 75,
        "relevance": 85,
    }
    base.update(overrides)
    return base


def check(data, range_start=0.0, range_end=600.0):
    return validate_candidate(data, range_start, range_end, DURATIONS)


def test_valid_candidate():
    candidate = check(raw())
    assert candidate.start == 60.0
    assert candidate.end == 120.0
    assert candidate.duration == 60.0
    assert candidate.title == "A title"
    assert candidate.scores["interestingness"] == 80
    assert candidate.missing_components == ()


def test_numeric_and_string_seconds():
    candidate = check(raw(start=75.5, end="130"))
    assert candidate.start == 75.5
    assert candidate.end == 130.0


def test_end_not_after_start_rejected():
    with pytest.raises(CandidateError):
        check(raw(start="0:02:00", end="0:01:00"))
    with pytest.raises(CandidateError):
        check(raw(start="0:02:00", end="0:02:00"))


def test_outside_range_rejected_but_small_overshoot_clamped():
    with pytest.raises(CandidateError):
        check(raw(start="0:09:00", end="0:11:40"))
    candidate = check(raw(start="0:09:00", end="0:10:03"))
    assert candidate.end == 600.0


def test_too_short_rejected():
    with pytest.raises(CandidateError):
        check(raw(start=60, end=63))


def test_too_long_rejected():
    with pytest.raises(CandidateError):
        validate_candidate(raw(start=0, end=900), 0.0, 2000.0, DURATIONS)


def test_clip_longer_than_absolute_max_is_allowed_for_later_trimming():
    candidate = validate_candidate(raw(start=0, end=300), 0.0, 2000.0, DURATIONS)
    assert candidate.duration == 300.0


def test_missing_interestingness_rejected():
    data = raw()
    del data["interestingness"]
    with pytest.raises(CandidateError):
        check(data)


def test_missing_other_scores_use_neutral_value_and_are_recorded():
    data = {"start": "0:01:00", "end": "0:02:00", "interestingness": 70}
    candidate = check(data)
    assert candidate.scores["hook_strength"] == NEUTRAL_SCORE
    assert "hook_strength" in candidate.missing_components
    assert "interestingness" not in candidate.missing_components
    assert candidate.title == "Untitled clip"


def test_scores_are_clamped_and_fractions_understood():
    candidate = check(raw(interestingness=150, hook_strength=-5, relevance="8/10"))
    assert candidate.scores["interestingness"] == 100.0
    assert candidate.scores["hook_strength"] == 0.0
    assert candidate.scores["relevance"] == pytest.approx(80.0)


def test_alias_keys_and_case_are_accepted():
    data = {
        "Start_Time": "0:01:00",
        "End Time": "0:02:00",
        "Name": "Aliased",
        "Why": "Because",
        "Interestingness_Score": 66,
    }
    data["end_time"] = data.pop("End Time")
    candidate = check(data)
    assert candidate.title == "Aliased"
    assert candidate.reason == "Because"
    assert candidate.scores["interestingness"] == 66


@pytest.mark.parametrize("bad", ["soon", True, None, [], -5, float("nan")])
def test_bad_time_values_rejected(bad):
    with pytest.raises(CandidateError):
        parse_time_value(bad)