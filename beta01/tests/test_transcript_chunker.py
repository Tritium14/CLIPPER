"""Tests for analysis/transcript_chunker.py."""

from analysis.transcript_chunker import chunk_transcript, format_segment_line
from input.transcript_models import Transcript, TranscriptSegment


def make_transcript(count):
    text = "word " * 10
    segments = [
        TranscriptSegment(i * 10.0, i * 10.0 + 10.0, f"{i} {text}".strip()) for i in range(count)
    ]
    return Transcript(segments, "text")


def test_line_format():
    segment = TranscriptSegment(3725.0, 3730.0, "hi")
    assert format_segment_line(segment) == "[1:02:05] hi"


def test_small_transcript_is_one_chunk():
    chunks = chunk_transcript(make_transcript(5), 10000, 500)
    assert len(chunks) == 1
    assert chunks[0].segment_count == 5
    assert chunks[0].start == 0.0
    assert chunks[0].end == 50.0


def test_empty_transcript_gives_no_chunks():
    assert chunk_transcript(Transcript([], "text"), 1000, 100) == []


def test_chunks_overlap_and_cover_everything():
    transcript = make_transcript(40)
    chunks = chunk_transcript(transcript, 400, 120)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 400.0
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start < previous.end  # they overlap
        assert following.start > previous.start  # and always make progress
    for segment in transcript.segments:
        line = format_segment_line(segment)
        assert any(line in chunk.text for chunk in chunks)


def test_oversized_segment_gets_its_own_chunk():
    segments = [TranscriptSegment(0.0, 5.0, "x" * 5000), TranscriptSegment(5.0, 10.0, "small")]
    chunks = chunk_transcript(Transcript(segments, "text"), 1000, 100)
    assert len(chunks) == 2
    assert chunks[0].segment_count == 1
    assert chunks[1].segment_count == 1