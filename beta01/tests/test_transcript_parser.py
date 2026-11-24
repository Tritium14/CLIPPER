"""Tests for input/transcript_parser.py."""

import pytest

from input.input_errors import TranscriptError
from input.transcript_parser import detect_format, load_transcript, parse_transcript_text

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello <i>there</i>,
welcome back.

2
00:00:04,000 --> 00:00:06,000
{\\an8}Second &amp; final.
"""

VTT = """WEBVTT
Kind: captions

NOTE this is a comment
that spans two lines

intro
00:00:01.000 --> 00:00:03.000 align:start position:0%
Hi <c.yellow>there</c>

00:03.500 --> 00:05.000
Next line
"""

PLAIN = """[00:00:01] First line
continues here
00:00:05 - Second
0:10
Third on its own line
"""


def test_detect_format():
    assert detect_format(SRT) == "srt"
    assert detect_format(VTT) == "vtt"
    assert detect_format(PLAIN) == "text"
    assert detect_format("\ufeffWEBVTT\n\n") == "vtt"


def test_parse_srt():
    transcript = parse_transcript_text(SRT)
    assert transcript.source_format == "srt"
    assert len(transcript.segments) == 2
    first, second = transcript.segments
    assert first.text == "Hello there, welcome back."
    assert first.start == pytest.approx(1.0)
    assert first.end == pytest.approx(3.5)
    assert not first.end_estimated
    assert second.text == "Second & final."


def test_parse_srt_with_windows_newlines():
    transcript = parse_transcript_text(SRT.replace("\n", "\r\n"))
    assert len(transcript.segments) == 2


def test_parse_vtt():
    transcript = parse_transcript_text(VTT)
    assert transcript.source_format == "vtt"
    assert [s.text for s in transcript.segments] == ["Hi there", "Next line"]
    assert transcript.segments[1].start == pytest.approx(3.5)


def test_parse_plain_text_variants():
    transcript = parse_transcript_text(PLAIN)
    assert transcript.source_format == "text"
    assert [s.text for s in transcript.segments] == [
        "First line continues here",
        "Second",
        "Third on its own line",
    ]
    assert [s.start for s in transcript.segments] == [1.0, 5.0, 10.0]
    assert transcript.segments[0].end == pytest.approx(5.0)
    assert transcript.segments[0].end_estimated
    assert transcript.segments[2].end > 10.0


def test_plain_text_with_explicit_range():
    transcript = parse_transcript_text("[00:01 - 00:04] hello world\n00:05 --> 00:08 bye")
    first, second = transcript.segments
    assert (first.start, first.end) == (1.0, 4.0)
    assert not first.end_estimated
    assert (second.start, second.end) == (5.0, 8.0)
    assert second.text == "bye"


def test_plain_text_malformed_timestamp_skipped():
    transcript = parse_transcript_text("00:99 bad line\n00:05 good line")
    assert [s.text for s in transcript.segments] == ["good line"]
    assert any("malformed" in w for w in transcript.warnings)


def test_srt_malformed_cue_skipped():
    text = "1\n00:00:xx --> 00:00:03,000\nbroken\n\n2\n00:00:04,000 --> 00:00:06,000\nok\n"
    transcript = parse_transcript_text(text, source_format="srt")
    assert [s.text for s in transcript.segments] == ["ok"]
    assert any("malformed" in w for w in transcript.warnings)


def test_srt_cue_without_blank_line_before_index():
    text = "1\n00:00:01,000 --> 00:00:02,000\none\n2\n00:00:03,000 --> 00:00:04,000\ntwo\n"
    transcript = parse_transcript_text(text)
    assert [s.text for s in transcript.segments] == ["one", "two"]


def test_text_before_first_timestamp_is_ignored():
    transcript = parse_transcript_text("Interview with Jane\n0:05 hello")
    assert [s.text for s in transcript.segments] == ["hello"]


def test_empty_and_untimed_transcripts_raise():
    with pytest.raises(TranscriptError):
        parse_transcript_text("   \n ")
    with pytest.raises(TranscriptError):
        parse_transcript_text("just some words with no timestamps")


def test_unsupported_format_raises():
    with pytest.raises(TranscriptError):
        parse_transcript_text(SRT, source_format="docx")


def test_video_duration_check_drops_late_segments():
    transcript = parse_transcript_text("0:05 early\n5:00 way too late", video_duration=60.0)
    assert [s.text for s in transcript.segments] == ["early"]
    assert any("different video" in w for w in transcript.warnings)


def test_all_segments_beyond_video_raises():
    with pytest.raises(TranscriptError):
        parse_transcript_text("5:00 late\n6:00 later", video_duration=60.0)


def test_load_missing_file_and_directory(tmp_path):
    with pytest.raises(TranscriptError):
        load_transcript(str(tmp_path / "nope.srt"))
    with pytest.raises(TranscriptError):
        load_transcript(str(tmp_path))
    with pytest.raises(TranscriptError):
        load_transcript("   ")


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.srt"
    path.write_bytes(b"   \n")
    with pytest.raises(TranscriptError):
        load_transcript(str(path))


def test_load_utf8_with_bom_and_quoted_path(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes("\ufeff0:01 café".encode("utf-8"))
    transcript = load_transcript(f'"{path}"')
    assert transcript.segments[0].text == "café"
    assert transcript.source_path == path.resolve()


def test_load_cp1252_file_warns(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes("0:01 café".encode("cp1252"))
    transcript = load_transcript(str(path))
    assert transcript.segments[0].text == "café"
    assert any("Windows-1252" in w for w in transcript.warnings)