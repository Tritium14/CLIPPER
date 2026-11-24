"""Exception types for the analysis modules."""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base class for problems talking to the AI provider."""


class LLMAuthError(LLMError):
    """The provider rejected the credentials or billing. Retrying will not help."""


class LLMModelError(LLMError):
    """This specific model cannot serve the request (not found, bad request)."""


class LLMTransientError(LLMError):
    """A temporary problem (rate limit, network, server error) that may pass."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ResponseParseError(Exception):
    """The model's reply did not contain usable JSON."""


class CandidateError(Exception):
    """An AI-suggested clip failed validation."""


class CandidateStoreError(Exception):
    """A candidates file could not be written or read."""


class HighlightDetectionError(Exception):
    """Highlight detection could not produce a result."""