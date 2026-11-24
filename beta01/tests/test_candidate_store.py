"""Tests for analysis/candidate_store.py and candidate_report.py."""

import pytest

from analysis.analysis_errors import CandidateStoreError
from analysis.candidate_report import format_candidate_table
from analysis.candidate_store import candidate_file_path, load_candidates, safe_name, save_candidates
from analysis.clip_models import SCORE_COMPONENTS, ClipCandidate


def make(start, end, title, interest=70.0):
    scores = {name: 50.0 for name in SCORE_COMPONENTS}
    scores["interestingness"] = interest
    return ClipCandidate(
        start=start,
        end=end,
        title=title,
        reason="because",
        scores=scores,
        category="story",
        chunk_index=2,
        missing_components=("relevance",),
    )


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "deep" / "c.json"
    original = [make(60.0, 120.0, "First"), make(300.5, 380.25, "Second", 90.0)]
    save_candidates(path, original, "My Video")
    loaded, metadata = load_candidates(path)
    assert loaded == original
    assert metadata["source"] == "My Video"
    assert metadata["created"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(CandidateStoreError):
        load_candidates(tmp_path / "nope.json")


def test_malformed_files_raise(tmp_path):
    bad_json = tmp_path / "a.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(CandidateStoreError):
        load_candidates(bad_json)

    bad_shape = tmp_path / "b.json"
    bad_shape.write_text('{"candidates": [{"start": 1}]}', encoding="utf-8")
    with pytest.raises(CandidateStoreError):
        load_candidates(bad_shape)

    bad_range = tmp_path / "c.json"
    save_candidates(bad_range, [make(10.0, 50.0, "ok")])
    bad_range.write_text(bad_range.read_text(encoding="utf-8").replace('"end": 50.0', '"end": 5.0'), encoding="utf-8")
    with pytest.raises(CandidateStoreError):
        load_candidates(bad_range)


def test_safe_name_and_path(tmp_path):
    assert safe_name("My Video: Part 1?") == "My_Video_Part_1"
    assert safe_name("???") == "source"
    path = candidate_file_path(tmp_path, "youtube_abc")
    assert path == tmp_path / "candidates" / "youtube_abc_candidates.json"


def test_report_ranks_by_interestingness():
    table = format_candidate_table(
        [make(10.0, 70.0, "Low", 40.0), make(100.0, 190.0, "High", 95.0)]
    )
    lines = table.splitlines()
    assert "High" in lines[1]
    assert "Low" in lines[2]
    assert "0:01:40-0:03:10" in lines[1]
    assert format_candidate_table([make(10.0, 70.0, "a"), make(80.0, 140.0, "b")], limit=1).count("\n") == 1