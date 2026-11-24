"""Data model for AI-suggested clip candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# The six subjective components the AI rates (0-100). Python combines them into
# the final weighted score in the clip scoring stage.
SCORE_COMPONENTS = (
    "interestingness",
    "hook_strength",
    "context_completeness",
    "emotional_impact",
    "standalone_quality",
    "relevance",
)


@dataclass(frozen=True)
class ClipCandidate:
    """A validated moment the AI suggested as a short clip."""

    start: float
    end: float
    title: str
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)
    category: str = ""
    chunk_index: int = 0
    # Components the AI did not provide; they were set to the neutral score.
    missing_components: Tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "title": self.title,
            "reason": self.reason,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "scores": dict(self.scores),
            "missing_components": list(self.missing_components),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipCandidate":
        """Rebuild a candidate. Raises KeyError, TypeError, or ValueError if malformed."""
        scores_raw = data["scores"]
        scores = {name: float(scores_raw[name]) for name in SCORE_COMPONENTS}
        start = float(data["start"])
        end = float(data["end"])
        if end <= start:
            raise ValueError("end must be greater than start")
        return cls(
            start=start,
            end=end,
            title=str(data["title"]),
            reason=str(data.get("reason", "")),
            scores=scores,
            category=str(data.get("category", "")),
            chunk_index=int(data.get("chunk_index", 0)),
            missing_components=tuple(str(x) for x in data.get("missing_components", [])),
        )