"""Tests for input/video_source_resolver.py."""

import pytest
from pathlib import Path

from config import AppConfig
from input import video_source_resolver as resolver
from input.input_errors import VideoInputError, VideoProbeError
from input.video_downloader import DownloadResult
from input.video_probe import VideoInfo


def _info(path):
    return VideoInfo(
        path=path,
        duration_seconds=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        rotation=0,
        size_bytes=1,
    )


@pytest.mark.parametrize(
    "text",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtu.be/dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtu.be/dQw4w9WgXcQ",
    ],
)
def test_looks_like_url_true(text):
    assert resolver.looks_like_url(text)


@pytest.mark.parametrize("text", ["C:\\videos\\talk.mp4", "/home/me/talk.mp4", "talk.mp4", ""])
def test_looks_like_url_false(text):
    assert not resolver.looks_like_url(text)


def test_normalize_url_adds_scheme():
    assert resolver.normalize_url("youtu.be/abc") == "https://youtu.be/abc"
    assert resolver.normalize_url("http://youtu.be/abc") == "http://youtu.be/abc"


def test_empty_input_rejected():
    with pytest.raises(VideoInputError):
        resolver.resolve_video_source("  ", AppConfig())


def test_local_path_resolves(tmp_path, monkeypatch):
    path = tmp_path / "talk.mp4"
    monkeypatch.setattr(resolver, "load_local_video", lambda text: _info(Path(text)))
    resolved = resolver.resolve_video_source(str(path), AppConfig())
    assert not resolved.is_remote
    assert resolved.source_name == "talk.mp4"
    assert resolved.source_url == ""
    assert "talk.mp4" in resolved.describe()


def test_url_resolves_via_downloader(tmp_path, monkeypatch):
    def fake_download(url, cache_dir, settings, progress_callback=None, force_redownload=False):
        return DownloadResult(
            path=tmp_path / "v.mp4",
            title="Great Talk",
            video_id="dQw4w9WgXcQ",
            source_url=url,
            platform="youtube",
            from_cache=False,
        )

    monkeypatch.setattr(resolver, "download_video", fake_download)
    monkeypatch.setattr(resolver, "probe_video", _info)
    resolved = resolver.resolve_video_source("youtu.be/dQw4w9WgXcQ", AppConfig())
    assert resolved.is_remote
    assert resolved.source_name == "Great Talk"
    assert resolved.source_url == "https://youtu.be/dQw4w9WgXcQ"


def test_corrupt_cache_triggers_redownload(tmp_path, monkeypatch):
    force_flags = []

    def fake_download(url, cache_dir, settings, progress_callback=None, force_redownload=False):
        force_flags.append(force_redownload)
        return DownloadResult(
            path=tmp_path / "v.mp4",
            title="T",
            video_id="dQw4w9WgXcQ",
            source_url=url,
            platform="youtube",
            from_cache=not force_redownload,
        )

    probe_calls = {"n": 0}

    def fake_probe(path):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            raise VideoProbeError("corrupt")
        return _info(path)

    monkeypatch.setattr(resolver, "download_video", fake_download)
    monkeypatch.setattr(resolver, "probe_video", fake_probe)
    resolved = resolver.resolve_video_source("youtu.be/dQw4w9WgXcQ", AppConfig())
    assert force_flags == [False, True]
    assert not resolved.from_cache


def test_probe_failure_on_fresh_download_propagates(tmp_path, monkeypatch):
    def fake_download(url, cache_dir, settings, progress_callback=None, force_redownload=False):
        return DownloadResult(
            path=tmp_path / "v.mp4",
            title="T",
            video_id="dQw4w9WgXcQ",
            source_url=url,
            platform="youtube",
            from_cache=False,
        )

    def failing_probe(path):
        raise VideoProbeError("bad file")

    monkeypatch.setattr(resolver, "download_video", fake_download)
    monkeypatch.setattr(resolver, "probe_video", failing_probe)
    with pytest.raises(VideoProbeError):
        resolver.resolve_video_source("youtu.be/dQw4w9WgXcQ", AppConfig())