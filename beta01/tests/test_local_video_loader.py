"""Tests for input/local_video_loader.py."""

import pytest

from input import local_video_loader
from input.input_errors import LocalVideoError
from input.video_probe import VideoInfo


def _fake_info(path):
    return VideoInfo(
        path=path,
        duration_seconds=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        rotation=0,
        size_bytes=1,
    )


def test_empty_path_rejected():
    with pytest.raises(LocalVideoError):
        local_video_loader.load_local_video("   ")


def test_missing_file_rejected(tmp_path):
    with pytest.raises(LocalVideoError):
        local_video_loader.load_local_video(str(tmp_path / "nope.mp4"))


def test_directory_rejected(tmp_path):
    with pytest.raises(LocalVideoError):
        local_video_loader.load_local_video(str(tmp_path))


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(LocalVideoError):
        local_video_loader.load_local_video(str(path))


def test_quoted_path_is_cleaned_and_loaded(tmp_path, monkeypatch):
    path = tmp_path / "clip.MP4"
    path.write_bytes(b"x")
    monkeypatch.setattr(local_video_loader, "probe_video", _fake_info)
    info = local_video_loader.load_local_video(f'  "{path}"  ')
    assert info.path.name == "clip.MP4"
    assert info.width == 1920