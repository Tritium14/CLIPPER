"""Reads video metadata using ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from input.input_errors import VideoProbeError

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 60


def format_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    total = int(round(max(seconds, 0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class VideoInfo:
    """Technical details of a video file.

    width and height are the *displayed* dimensions (rotation already applied).
    """

    path: Path
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str]
    rotation: int
    size_bytes: int

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def describe(self) -> str:
        """One-line human-readable summary."""
        audio = self.audio_codec or "none"
        size_mb = self.size_bytes / (1024 * 1024)
        return (
            f"{self.path.name}: {format_hms(self.duration_seconds)} | "
            f"{self.width}x{self.height} @ {self.fps:.2f} fps | "
            f"video {self.video_codec} | audio {audio} | {size_mb:.1f} MB"
        )


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_fraction(value: Any) -> float:
    """Parse ffprobe frame rates such as '30000/1001'. Returns 0.0 if unusable."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _stream_rotation(stream: Dict[str, Any]) -> int:
    """Return the stream rotation in degrees (0, 90, 180, or 270)."""
    rotation: Any = (stream.get("tags") or {}).get("rotate")
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = side_data["rotation"]
    if rotation is None:
        return 0
    try:
        return int(float(rotation)) % 360
    except (TypeError, ValueError):
        return 0


def parse_probe_output(data: Dict[str, Any], path: Path, size_bytes: int = 0) -> VideoInfo:
    """Convert ffprobe JSON output into a VideoInfo.

    Raises:
        VideoProbeError: if there is no usable video stream or duration.
    """
    streams = data.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise VideoProbeError(f"No video stream found in '{path.name}'.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width <= 0 or height <= 0:
        raise VideoProbeError(f"Could not read the frame size of '{path.name}'.")

    rotation = _stream_rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    duration = _parse_float((data.get("format") or {}).get("duration"))
    if duration <= 0:
        duration = _parse_float(video.get("duration"))
    if duration <= 0:
        raise VideoProbeError(f"Could not determine the duration of '{path.name}'.")

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(video.get("r_frame_rate"))

    return VideoInfo(
        path=path,
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name") or "unknown") if audio else None,
        rotation=rotation,
        size_bytes=size_bytes,
    )


def probe_video(path: Path) -> VideoInfo:
    """Run ffprobe on a file and return its VideoInfo.

    Raises:
        VideoProbeError: if ffprobe is missing, fails, or the file is unusable.
    """
    path = Path(path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoProbeError("ffprobe was not found on PATH. Install FFmpeg first.")

    command = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    logger.debug("Running: %s", " ".join(command))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoProbeError(f"ffprobe timed out reading '{path.name}'.") from exc
    except OSError as exc:
        raise VideoProbeError(f"Could not run ffprobe: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip()[:500] or "unknown error"
        raise VideoProbeError(f"ffprobe could not read '{path.name}': {detail}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"ffprobe returned unreadable output for '{path.name}'.") from exc

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return parse_probe_output(data, path, size_bytes)