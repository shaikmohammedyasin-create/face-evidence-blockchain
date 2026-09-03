"""
app/search/candidates.py — Candidate data structures and URL deduplication utilities.
"""
from __future__ import annotations

from app.face.matcher import deduplicate_candidates, normalize_url
from app.models.schemas import SearchCandidate

__all__ = [
    "SearchCandidate",
    "normalize_url",
    "deduplicate_candidates",
]
