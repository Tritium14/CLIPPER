"""Tests for input/video_probe.py (parsing only; no ffprobe needed)."""

from pathlib import Path

import pytest

from input.input_errors import VideoProbeError
from input.video_probe import format_hms, parse_probe_output

PATH = Path("sample.mp4")


def _data(**video_overrides):
    video = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001",
    }
    video.update(video_overrides)
    return {
        "streams": [video, {"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": "125.5"},
    }


def test_basic_parse():
    info = parse_probe_output(_data(), PATH, size_bytes=1000)
    assert (info.width, info.height) == (1920, 1080)
    assert info.duration_seconds == pytest.approx(125.5)
    assert info.fps == pytest.approx(29.97, abs=0.01)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.has_audio
    assert info.size_bytes == 1000


def test_rotation_side_data_swaps_dimensions():
    data = _data(side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}])
    info = parse_probe_output(data, PATH)
    assert (info.width, info.height) == (1080, 1920)
    assert info.rotation == 270


def test_rotate_tag_swaps_dimensions():
    info = parse_probe_output(_data(tags={"rotate": "90"}), PATH)
    assert (info.width, info.height) == (1080, 1920)


def test_no_audio_stream():
    data = _data()
    data["streams"] = data["streams"][:1]
    info = parse_probe_output(data, PATH)
    assert info.audio_codec is None
    assert not info.has_audio


def test_no_video_stream_raises():
    with pytest.raises(VideoProbeError):
        parse_probe_output({"streams": [{"codec_type": "audio"}], "format": {"duration": "5"}}, PATH)


def test_cover_art_is_not_treated_as_video():
    data = _data()
    cover = {
        "codec_type": "video",
        "codec_name": "mjpeg",
        "width": 500,
        "height": 500,
        "disposition": {"attached_pic": 1},
    }
    data["streams"].insert(0, cover)
    info = parse_probe_output(data, PATH)
    assert info.video_codec == "h264"


def test_zero_duration_raises():
    data = _data()
    data["format"] = {"duration": "0"}
    with pytest.raises(VideoProbeError):
        parse_probe_output(data, PATH)


def test_duration_falls_back_to_stream():
    data = _data(duration="42.0")
    data["format"] = {}
    assert parse_probe_output(data, PATH).duration_seconds == pytest.approx(42.0)


def test_fps_falls_back_to_r_frame_rate():
    info = parse_probe_output(_data(avg_frame_rate="0/0", r_frame_rate="25/1"), PATH)
    assert info.fps == pytest.approx(25.0)


def test_invalid_frame_size_raises():
    with pytest.raises(VideoProbeError):
        parse_probe_output(_data(width=0), PATH)


def test_format_hms():
    assert format_hms(0) == "0:00:00"
    assert format_hms(3725) == "1:02:05"