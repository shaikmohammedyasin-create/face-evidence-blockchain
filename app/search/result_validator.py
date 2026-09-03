"""
app/search/result_validator.py — Validation helpers for search candidates.

Distinguishes:
    SearchCandidate   — URL returned by the search engine (not yet verified)
    MatchResult       — after face comparison with confidence ≥ threshold
"""
from __future__ import annotations

from app.models.schemas import MatchResult, SearchCandidate
from app.utils.logger import get_logger

log = get_logger(__name__)


def filter_accessible_candidates(
    candidates: list[SearchCandidate],
) -> list[SearchCandidate]:
    """
    Drop candidates whose URLs are obviously inaccessible or unsuitable
    for image download/comparison.

    We do NOT fetch pages here — just lightweight URL-level checks.
    """
    import re

    # Patterns that indicate login-walls, media redirectors, or non-image pages
    _SKIP_PATTERNS = [
        r"\.pdf$",
        r"\.docx?$",
        r"accounts\.google\.com",
        r"login\.",
        r"/auth/",
        r"signup",
        r"javascript:void",
    ]
    compiled = [re.compile(p, re.I) for p in _SKIP_PATTERNS]

    valid = []
    for c in candidates:
        skip = any(pat.search(c.url) for pat in compiled)
        if skip:
            log.debug("Skipping inaccessible URL: %s", c.url[:80])
        else:
            valid.append(c)

    log.info(
        "Candidate filter: %d → %d (dropped %d)",
        len(candidates), len(valid), len(candidates) - len(valid),
    )
    return valid


def summarise_matches(results: list[MatchResult]) -> dict:
    """Return a summary dict suitable for logging / display."""
    matched = [r for r in results if r.matched]
    return {
        "total_candidates": len(results),
        "matched": len(matched),
        "best_confidence": round(max((r.confidence for r in results), default=0.0), 4),
        "top_match_url": matched[0].candidate.url if matched else None,
        "top_match_platform": matched[0].candidate.platform if matched else None,
    }
