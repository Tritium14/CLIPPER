"""Tests for input/video_downloader.py (no network or yt-dlp needed)."""

import json

import pytest

from config import DownloadConfig
from input.input_errors import UnsupportedSourceError
from input.video_downloader import (
    download_video,
    find_cached_video,
    validate_url,
)

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=42",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PL123456",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"  https://www.youtube.com/embed/{VIDEO_ID}  ",
    ],
)
def test_valid_youtube_urls(url):
    result = validate_url(url)
    assert result.platform == "youtube"
    assert result.video_id == VIDEO_ID
    assert result.cache_prefix == f"youtube_{VIDEO_ID}"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://vimeo.com/12345",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/",
        f"ftp://youtube.com/watch?v={VIDEO_ID}",
    ],
)
def test_invalid_urls_rejected(url):
    with pytest.raises(UnsupportedSourceError):
        validate_url(url)


def test_find_cached_video_missing_folder(tmp_path):
    assert find_cached_video(tmp_path / "nope", "youtube_abc") is None


def test_find_cached_video_finds_complete_file(tmp_path):
    (tmp_path / "youtube_abc.mp4").write_bytes(b"data")
    found = find_cached_video(tmp_path, "youtube_abc")
    assert found is not None and found.name == "youtube_abc.mp4"


def test_find_cached_video_ignores_partial_and_intermediate(tmp_path):
    (tmp_path / "youtube_abc.mp4.part").write_bytes(b"data")
    (tmp_path / "youtube_abc.f137.mp4").write_bytes(b"data")
    (tmp_path / "youtube_abc.meta.json").write_text("{}")
    assert find_cached_video(tmp_path, "youtube_abc") is None


def test_find_cached_video_ignores_empty_file(tmp_path):
    (tmp_path / "youtube_abc.mp4").write_bytes(b"")
    assert find_cached_video(tmp_path, "youtube_abc") is None


def test_download_uses_cache_without_network(tmp_path):
    prefix = f"youtube_{VIDEO_ID}"
    (tmp_path / f"{prefix}.mp4").write_bytes(b"video")
    (tmp_path / f"{prefix}.meta.json").write_text(json.dumps({"title": "My Talk"}), encoding="utf-8")
    result = download_video(
        f"https://www.youtube.com/watch?v={VIDEO_ID}", tmp_path, DownloadConfig()
    )
    assert result.from_cache
    assert result.title == "My Talk"
    assert result.video_id == VIDEO_ID
    assert result.path.name == f"{prefix}.mp4"


def test_download_config_validation():
    from config import ConfigError

    with pytest.raises(ConfigError):
        DownloadConfig(max_height=10).validate()
    with pytest.raises(ConfigError):
        DownloadConfig(merge_format="avi").validate()
    DownloadConfig().validate()