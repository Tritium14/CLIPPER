"""Formats clip candidates as a readable text table."""

from __future__ import annotations

from typing import List, Optional

from analysis.clip_models import ClipCandidate
from input.timestamp_utils import format_timestamp


def format_candidate_table(candidates: List[ClipCandidate], limit: Optional[int] = None) -> str:
    """Rank candidates by interestingness and format them as a table."""
    ranked = sorted(candidates, key=lambda c: (-c.scores.get("interestingness", 0.0), c.start))
    if limit is not None:
        ranked = ranked[:limit]

    lines = [f"{'#':>3}  {'Time':<17} {'Len':>5}  {'Int':>3} {'Hook':>4} {'Ctx':>3}  Title"]
    for rank, candidate in enumerate(ranked, start=1):
        span = f"{format_timestamp(candidate.start)}-{format_timestamp(candidate.end)}"
        title = candidate.title if len(candidate.title) <= 60 else candidate.title[:57] + "..."
        lines.append(
            f"{rank:>3}  {span:<17} {candidate.duration:>4.0f}s  "
            f"{candidate.scores.get('interestingness', 0):>3.0f} "
            f"{candidate.scores.get('hook_strength', 0):>4.0f} "
            f"{candidate.scores.get('context_completeness', 0):>3.0f}  {title}"
        )
    return "\n".join(lines)