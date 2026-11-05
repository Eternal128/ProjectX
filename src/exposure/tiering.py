"""Assign each benchmark item to an exposure tier.

* ``confirmed``: a long exact match covering both question and answer.
* ``likely``: high semantic similarity with no exact match.
* ``clean``: no candidate above a low similarity threshold.
* ``ambiguous``: everything else; excluded from both the exposed and clean groups.

Missed exposure puts seen items in the clean group, which biases effects toward
zero. State this in every write-up.
"""
from __future__ import annotations

from dataclasses import dataclass

CONFIRMED, LIKELY, CLEAN, AMBIGUOUS = "confirmed", "likely", "clean", "ambiguous"
TIERS = (CONFIRMED, LIKELY, CLEAN, AMBIGUOUS)


@dataclass(frozen=True)
class TierThresholds:
    likely_min_similarity: float
    clean_max_similarity: float

    def __post_init__(self) -> None:
        if not self.clean_max_similarity < self.likely_min_similarity:
            raise ValueError("clean_max_similarity must be below likely_min_similarity")

    @classmethod
    def from_config(cls, cfg: dict) -> "TierThresholds":
        t = cfg["tiers"]
        return cls(t["likely_min_similarity"], t["clean_max_similarity"])


def assign_tier(qa_exact_match: bool, max_similarity: float | None, th: TierThresholds) -> str:
    """Tier for one item.

    ``qa_exact_match`` is True when a long exact match covers both question and
    answer. ``max_similarity`` is the highest cosine similarity over candidate
    passages, or None when retrieval returned no candidates.
    """
    if qa_exact_match:
        return CONFIRMED
    sim = -1.0 if max_similarity is None else float(max_similarity)
    if sim >= th.likely_min_similarity:
        return LIKELY
    if sim < th.clean_max_similarity:
        return CLEAN
    return AMBIGUOUS
