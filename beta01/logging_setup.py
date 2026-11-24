"""Logging configuration with automatic secret redaction.

Any environment variable whose name contains KEY, TOKEN, or SECRET has its value
masked in all log output, so API keys cannot leak into the console or log files.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE_NAME = "intelligent_shorts.log"
_SENSITIVE_NAME_PARTS = ("KEY", "TOKEN", "SECRET")
_MIN_SECRET_LENGTH = 8


def _current_secrets() -> List[str]:
    """Collect secret values from the environment (re-read on every call)."""
    secrets = []
    for name, value in os.environ.items():
        if any(part in name.upper() for part in _SENSITIVE_NAME_PARTS):
            value = value.strip()
            if len(value) >= _MIN_SECRET_LENGTH:
                secrets.append(value)
    return secrets


class SecretRedactingFilter(logging.Filter):
    """Replace known secret values in log messages with ***."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _current_secrets()
        if not secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in secrets:
            redacted = redacted.replace(secret, "***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def redact_text(text: str) -> str:
    """Mask known secrets in arbitrary text (for example an error message)."""
    for secret in _current_secrets():
        text = text.replace(secret, "***")
    return text


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logging to the console and a rotating log file.

    Safe to call more than once; existing handlers are replaced.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)
    redactor = SecretRedactingFilter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(redactor)
    root.addHandler(console)

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / LOG_FILE_NAME,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Could not create log file in '%s': %s", log_dir, exc)