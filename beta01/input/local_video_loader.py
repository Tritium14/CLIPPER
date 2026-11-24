"""Loads and validates a video file chosen from the local computer."""

from __future__ import annotations

import logging
from pathlib import Path

from input.input_errors import LocalVideoError
from input.video_probe import VideoInfo, probe_video

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")


def clean_path_input(text: str) -> str:
    """Strip whitespace and the quotes Windows adds when copying a path."""
    return (text or "").strip().strip('"').strip("'").strip()


def load_local_video(path_text: str) -> VideoInfo:
    """Validate a local video path and read its metadata.

    Raises:
        LocalVideoError: if the path is empty, missing, not a file, or has an
            unsupported extension.
        VideoProbeError: if the file cannot be read as a video.
    """
    cleaned = clean_path_input(path_text)
    if not cleaned:
        raise LocalVideoError("No video file path was provided.")

    path = Path(cleaned).expanduser()
    if not path.exists():
        raise LocalVideoError(f"Video file not found: {path}")
    if not path.is_file():
        raise LocalVideoError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise LocalVideoError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    info = probe_video(path.resolve())
    logger.info("Loaded local video: %s", info.describe())
    return info