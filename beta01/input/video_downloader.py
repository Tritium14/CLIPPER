"""Downloads videos from supported sites using yt-dlp.

This module only downloads. Reading metadata and processing video happen in
other modules. Downloads are cached by site and video ID, so the same source is
never downloaded twice.

To support another site, add a PlatformHandler subclass and append an instance
to PLATFORM_HANDLERS.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from config import DownloadConfig
from input.input_errors import UnsupportedSourceError, VideoDownloadError

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")
_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Called with (percent 0-100, status text) while downloading.
ProgressCallback = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Platform handlers
# --------------------------------------------------------------------------- #
class PlatformHandler(ABC):
    """Knows how to recognize one video site and extract a video ID."""

    name: str = ""

    @abstractmethod
    def matches(self, url: str) -> bool:
        """True if the URL belongs to this site."""

    @abstractmethod
    def extract_video_id(self, url: str) -> Optional[str]:
        """Return the video ID, or None if the URL has none."""


class YouTubeHandler(PlatformHandler):
    """YouTube watch, youtu.be, shorts, embed, and live URLs."""

    name = "youtube"
    _HOSTS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
    _PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")

    def matches(self, url: str) -> bool:
        return (urlparse(url).hostname or "").lower() in self._HOSTS

    def extract_video_id(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path
        candidate: Optional[str] = None

        if host.endswith("youtu.be"):
            candidate = path.lstrip("/").split("/")[0]
        elif path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            for prefix in self._PATH_PREFIXES:
                if path.startswith(prefix):
                    candidate = path[len(prefix):].split("/")[0]
                    break

        if candidate and _YOUTUBE_ID.match(candidate):
            return candidate
        return None


PLATFORM_HANDLERS: List[PlatformHandler] = [YouTubeHandler()]


@dataclass(frozen=True)
class ValidatedUrl:
    """A URL confirmed to belong to a supported site."""

    platform: str
    video_id: str
    url: str

    @property
    def cache_prefix(self) -> str:
        return f"{self.platform}_{self.video_id}"


def validate_url(url: str) -> ValidatedUrl:
    """Check that a URL is a supported video URL.

    Raises:
        UnsupportedSourceError: if the URL is empty, malformed, from an
            unsupported site, or has no video ID.
    """
    url = (url or "").strip()
    if not url:
        raise UnsupportedSourceError("The video URL is empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise UnsupportedSourceError(f"'{url}' is not a valid http(s) URL.")

    for handler in PLATFORM_HANDLERS:
        if handler.matches(url):
            video_id = handler.extract_video_id(url)
            if not video_id:
                raise UnsupportedSourceError(
                    f"Could not find a video ID in this {handler.name} URL: {url}"
                )
            return ValidatedUrl(handler.name, video_id, url)

    supported = ", ".join(h.name for h in PLATFORM_HANDLERS)
    raise UnsupportedSourceError(f"Unsupported video site. Supported sites: {supported}.")


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DownloadResult:
    """A video file ready for processing."""

    path: Path
    title: str
    video_id: str
    source_url: str
    platform: str
    from_cache: bool


def find_cached_video(cache_dir: Path, prefix: str) -> Optional[Path]:
    """Find a fully downloaded video named '<prefix>.<ext>' in the cache folder.

    Partial downloads and yt-dlp's unmerged intermediate files are ignored.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
    candidates = []
    for path in cache_dir.glob(f"{prefix}.*"):
        if (
            path.stem == prefix
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.is_file()
            and path.stat().st_size > 0
        ):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _meta_path(cache_dir: Path, prefix: str) -> Path:
    return Path(cache_dir) / f"{prefix}.meta.json"


def _write_meta(cache_dir: Path, prefix: str, meta: Dict[str, Any]) -> None:
    try:
        _meta_path(cache_dir, prefix).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not write download metadata: %s", exc)


def _read_title(cache_dir: Path, prefix: str, fallback: str) -> str:
    try:
        data = json.loads(_meta_path(cache_dir, prefix).read_text(encoding="utf-8"))
        title = str(data.get("title") or "").strip()
        return title or fallback
    except (OSError, json.JSONDecodeError, AttributeError):
        return fallback


def _remove_cached(cache_dir: Path, prefix: str) -> None:
    """Delete every cached file for a video (used to force a clean re-download)."""
    for path in Path(cache_dir).glob(f"{prefix}.*"):
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# yt-dlp integration
# --------------------------------------------------------------------------- #
class _YtDlpLogger:
    """Routes yt-dlp messages into the application logger."""

    def debug(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        logger.warning("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        logger.error("yt-dlp: %s", message)


def _make_progress_hook(callback: Optional[ProgressCallback]) -> Callable[[Dict[str, Any]], None]:
    state = {"last_logged": -10.0}

    def hook(data: Dict[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            done = data.get("downloaded_bytes") or 0
            if total:
                percent = min(done / total * 100.0, 100.0)
                if callback:
                    callback(percent, "downloading")
                if percent - state["last_logged"] >= 10.0:
                    logger.info("Downloading... %.0f%%", percent)
                    state["last_logged"] = percent
        elif status == "finished":
            state["last_logged"] = -10.0
            if callback:
                callback(100.0, "download finished")
            logger.info("Download stream finished (merging if needed).")

    return hook


def _explain_download_error(exc: Exception) -> str:
    """Turn a yt-dlp error into a clear message with a hint when possible."""
    message = str(exc).replace("ERROR: ", "").strip()
    lowered = message.lower()
    hint = ""
    if any(t in lowered for t in ("sign in", "not a bot", "private video", "age", "members-only")):
        hint = (
            " This video may need a logged-in session. Export your browser cookies to "
            "a file and set downloads.cookies_file in config.json."
        )
    elif any(
        t in lowered
        for t in ("getaddrinfo", "timed out", "temporary failure", "connection", "network")
    ):
        hint = " Check your internet connection and try again."
    elif "unavailable" in lowered or "removed" in lowered:
        hint = " The video may have been removed or restricted in your region."
    return f"Download failed: {message}{hint}"


def _path_from_info(info: Dict[str, Any], cache_dir: Path, prefix: str) -> Optional[Path]:
    for item in info.get("requested_downloads") or []:
        filepath = item.get("filepath")
        if filepath and Path(filepath).is_file():
            return Path(filepath)
    return find_cached_video(cache_dir, prefix)


def download_video(
    url: str,
    cache_dir: Path,
    settings: DownloadConfig,
    progress_callback: Optional[ProgressCallback] = None,
    force_redownload: bool = False,
) -> DownloadResult:
    """Download a video (or reuse a cached copy).

    Args:
        url: A supported video URL.
        cache_dir: Folder where source videos are stored.
        settings: Download settings.
        progress_callback: Optional function called with (percent, status).
        force_redownload: Delete any cached copy and download again.

    Raises:
        UnsupportedSourceError: for invalid or unsupported URLs.
        VideoDownloadError: if the download fails.
    """
    validated = validate_url(url)
    cache_dir = Path(cache_dir)
    prefix = validated.cache_prefix

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoDownloadError(f"Cannot create download folder '{cache_dir}': {exc}") from exc

    if force_redownload:
        _remove_cached(cache_dir, prefix)
    elif settings.reuse_existing:
        cached = find_cached_video(cache_dir, prefix)
        if cached:
            title = _read_title(cache_dir, prefix, cached.stem)
            logger.info("Using cached download: %s", cached.name)
            return DownloadResult(
                path=cached,
                title=title,
                video_id=validated.video_id,
                source_url=validated.url,
                platform=validated.platform,
                from_cache=True,
            )

    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoDownloadError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

    height = settings.max_height
    options: Dict[str, Any] = {
        "format": f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "merge_output_format": settings.merge_format,
        "outtmpl": str(cache_dir / f"{prefix}.%(ext)s"),
        "noplaylist": True,
        "retries": settings.retries,
        "socket_timeout": settings.socket_timeout_seconds,
        "quiet": True,
        "noprogress": True,
        "logger": _YtDlpLogger(),
        "progress_hooks": [_make_progress_hook(progress_callback)],
    }
    if settings.cookies_file:
        cookies = Path(settings.cookies_file).expanduser()
        if not cookies.is_file():
            raise VideoDownloadError(f"Cookies file not found: {cookies}")
        options["cookiefile"] = str(cookies)

    logger.info("Downloading %s video %s (up to %dp)...", validated.platform, validated.video_id, height)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(validated.url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise VideoDownloadError(_explain_download_error(exc)) from exc
    except OSError as exc:
        raise VideoDownloadError(f"File error while downloading: {exc}") from exc

    if not info:
        raise VideoDownloadError("Download finished but no video information was returned.")

    path = _path_from_info(info, cache_dir, prefix)
    if path is None:
        raise VideoDownloadError("Download finished but the video file could not be found.")

    title = str(info.get("title") or path.stem)
    _write_meta(
        cache_dir,
        prefix,
        {
            "title": title,
            "url": validated.url,
            "video_id": validated.video_id,
            "platform": validated.platform,
            "uploader": info.get("uploader") or "",
        },
    )
    logger.info("Downloaded: %s", path.name)
    return DownloadResult(
        path=path,
        title=title,
        video_id=validated.video_id,
        source_url=validated.url,
        platform=validated.platform,
        from_cache=False,
    )