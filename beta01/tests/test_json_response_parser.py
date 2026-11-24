"""Tests for analysis/json_response_parser.py."""

import pytest

from analysis.analysis_errors import ResponseParseError
from analysis.json_response_parser import extract_clip_dicts

CLIP_A = '{"start": "0:01:00", "end": "0:02:00", "title": "A"}'
CLIP_B = '{"start": "0:05:00", "end": "0:06:00", "title": "B"}'


def titles(items):
    return [item["title"] for item in items]


def test_plain_array():
    assert titles(extract_clip_dicts(f"[{CLIP_A},{CLIP_B}]")) == ["A", "B"]


def test_code_fence():
    assert titles(extract_clip_dicts(f"```json\n[{CLIP_A}]\n```")) == ["A"]


def test_think_block_is_removed():
    reply = f"<think>maybe [ {{bad}} ] hmm</think>\n[{CLIP_A}]"
    assert titles(extract_clip_dicts(reply)) == ["A"]


def test_wrapper_object():
    assert titles(extract_clip_dicts(f'{{"clips": [{CLIP_A}]}}')) == ["A"]


def test_trailing_comma():
    assert titles(extract_clip_dicts(f"[{CLIP_A},]")) == ["A"]


def test_text_around_json():
    reply = f"Here are the best moments:\n[{CLIP_A}, {CLIP_B}]\nHope that helps!"
    assert titles(extract_clip_dicts(reply)) == ["A", "B"]


def test_truncated_reply_salvages_complete_objects():
    reply = f'[{CLIP_A},{CLIP_B},{{"start": "0:09:00", "end": "0:1'
    assert titles(extract_clip_dicts(reply)) == ["A", "B"]


def test_single_object():
    assert titles(extract_clip_dicts(CLIP_A)) == ["A"]


def test_braces_inside_strings_do_not_break_scanning():
    clip = '{"start": "0:01:00", "end": "0:02:00", "title": "Use {braces} and \\"quotes\\""}'
    reply = f"Sure! {clip} and that is all"
    assert extract_clip_dicts(reply)[0]["title"] == 'Use {braces} and "quotes"'


def test_empty_array_is_a_valid_answer():
    assert extract_clip_dicts("[]") == []
    assert extract_clip_dicts('{"clips": []}') == []


@pytest.mark.parametrize("reply", ["", "   ", "Sorry, I cannot help with that.", "[1, 2, 3]"])
def test_unusable_replies_raise(reply):
    with pytest.raises(ResponseParseError):
        extract_clip_dicts(reply)