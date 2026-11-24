"""Tests for input/timestamp_utils.py."""

import pytest

from input.input_errors import TimestampError
from input.timestamp_utils import format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    "text,expected",
    [
        ("00:01:23,456", 83.456),
        ("00:01:23.456", 83.456),
        ("1:02:03", 3723.0),
        ("01:23.5", 83.5),
        ("12:34", 754.0),
        ("125:30", 7530.0),
        ("[00:05]", 5.0),
        ("(0:10)", 10.0),
        ("90s", 90.0),
        ("123.5s", 123.5),
        ("1h2m3s", 3723.0),
    ],
)
def test_parse_valid(text, expected):
    assert parse_timestamp(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "   ", "abc", "00:61", "1:75:00", "12", "00:00:00:00", "-5"])
def test_parse_invalid(text):
    with pytest.raises(TimestampError):
        parse_timestamp(text)


def test_format_timestamp():
    assert format_timestamp(0) == "0:00:00"
    assert format_timestamp(3725) == "1:02:05"
    assert format_timestamp(-4) == "0:00:00"
    assert format_timestamp(3725.5, millis=True) == "1:02:05.500"
    assert format_timestamp(0.007, millis=True) == "0:00:00.007"


def test_round_trip():
    assert parse_timestamp(format_timestamp(4321.25, millis=True)) == pytest.approx(4321.25)