"""Turns what the user typed (a file path or a URL) into a ready-to-use video."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import AppConfig
from input.input_errors import VideoInputError, VideoProbeError
from input.local_video_loader import clean_path_input, load_local_video
from input.video_downloader import ProgressCallback, download_video
from input.video_probe import VideoInfo, probe_video

logger = logging.getLogger(__name__)

_URL_PREFIXES = ("http://", "https://", "www.")
_BARE_HOST_PREFIXES = ("youtube.com/", "m.youtube.com/", "youtu.be/")


@dataclass(frozen=True)
class ResolvedVideo:
    """A source video ready for the rest of the pipeline."""

    info: VideoInfo
    source_name: str  # video title for URLs, file name for local files
    source_url: str  # empty for local files
    from_cache: bool
    is_remote: bool

    @property
    def path(self) -> Path:
        return self.info.path

    def describe(self) -> str:
        """Multi-line summary for display."""
        lines = [f"Source:   {self.source_name}"]
        if self.source_url:
            lines.append(f"URL:      {self.source_url}")
        origin = "cached download" if self.from_cache else ("downloaded" if self.is_remote else "local file")
        lines.append(f"Path:     {self.path} ({origin})")
        lines.append(f"Details:  {self.info.describe()}")
        return "\n".join(lines)


def looks_like_url(text: str) -> bool:
    """True if the text looks like a web address rather than a file path."""
    lowered = (text or "").strip().lower()
    return lowered.startswith(_URL_PREFIXES) or lowered.startswith(_BARE_HOST_PREFIXES)


def normalize_url(text: str) -> str:
    """Add https:// to an address typed without a scheme."""
    cleaned = text.strip()
    if not cleaned.lower().startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    return cleaned


def _resolve_url(text: str, config: AppConfig, progress_callback: Optional[ProgressCallback]) -> ResolvedVideo:
    url = normalize_url(text)
    cache_dir = config.paths.data_dir_path / config.downloads.cache_dir

    result = download_video(url, cache_dir, config.downloads, progress_callback)
    try:
        info = probe_video(result.path)
    except VideoProbeError as exc:
        if not result.from_cache:
            raise
        logger.warning("Cached download is unreadable (%s). Downloading it again.", exc)
        result = download_video(
            url, cache_dir, config.downloads, progress_callback, force_redownload=True
        )
        info = probe_video(result.path)

    return ResolvedVideo(
        info=info,
        source_name=result.title,
        source_url=result.source_url,
        from_cache=result.from_cache,
        is_remote=True,
    )


def resolve_video_source(
    user_input: str,
    config: AppConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> ResolvedVideo:
    """Resolve a local path or URL into a validated, probed video.

    Raises:
        VideoInputError (or a subclass): for any input, download, or probe problem.
    """
    text = clean_path_input(user_input)
    if not text:
        raise VideoInputError("No video was provided. Enter a file path or a video URL.")

    if looks_like_url(text):
        return _resolve_url(text, config, progress_callback)

    info = load_local_video(text)
    return ResolvedVideo(
        info=info,
        source_name=info.path.name,
        source_url="",
        from_cache=False,
        is_remote=False,
    )