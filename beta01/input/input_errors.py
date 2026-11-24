"""Exception types for the input modules (video and transcript)."""

from __future__ import annotations


class VideoInputError(Exception):
    """Base class for all video input problems."""


class VideoProbeError(VideoInputError):
    """ffprobe could not read or understand a video file."""


class LocalVideoError(VideoInputError):
    """A local video file is missing, unsupported, or unusable."""


class UnsupportedSourceError(VideoInputError):
    """A URL is malformed or belongs to an unsupported site."""


class VideoDownloadError(VideoInputError):
    """Downloading a video failed."""


class TranscriptError(Exception):
    """A transcript is missing, unreadable, empty, or unusable."""


class TimestampError(TranscriptError):
    """A timestamp could not be parsed."""