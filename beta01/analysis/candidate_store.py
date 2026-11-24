"""Saves and loads clip candidates as JSON so detection results can be reused."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from analysis.analysis_errors import CandidateStoreError
from analysis.clip_models import ClipCandidate

logger = logging.getLogger(__name__)


def safe_name(name: str) -> str:
    """Turn any text into a safe file name stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._-")[:80]
    return cleaned or "source"


def candidate_file_path(data_dir: Path, name: str) -> Path:
    """Where the candidates for a source are stored: <data_dir>/candidates/<name>_candidates.json."""
    return Path(data_dir) / "candidates" / f"{safe_name(name)}_candidates.json"


def save_candidates(path: Path, candidates: List[ClipCandidate], source_name: str = "") -> None:
    """Write candidates to a JSON file (written atomically).

    Raises:
        CandidateStoreError: if the file cannot be written.
    """
    path = Path(path)
    payload: Dict[str, Any] = {
        "source": source_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates": [c.to_dict() for c in candidates],
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        raise CandidateStoreError(f"Cannot write candidates file '{path}': {exc}") from exc
    logger.info("Saved %d candidate(s) to %s", len(candidates), path)


def load_candidates(path: Path) -> Tuple[List[ClipCandidate], Dict[str, Any]]:
    """Read candidates from a JSON file. Returns (candidates, metadata).

    Raises:
        CandidateStoreError: if the file is missing, unreadable, or malformed.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CandidateStoreError(f"Candidates file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateStoreError(f"Cannot read candidates file '{path}': {exc}") from exc

    try:
        candidates = [ClipCandidate.from_dict(item) for item in data["candidates"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateStoreError(f"Candidates file '{path}' is malformed: {exc}") from exc
    metadata = {"source": data.get("source", ""), "created": data.get("created", "")}
    return candidates, metadata