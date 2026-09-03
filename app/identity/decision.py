"""
app/identity/decision.py — Three-tier calibrated forensic identity decision engine.

States:
  - HIGH_CONFIDENCE_MATCH (Tier: HIGH_MATCH, is_match=True):
      Similarity >= 0.440, runner-up margin >= 0.035, quality verified.
  - POSSIBLE_MATCH_REVIEW (Tier: REVIEW, is_match=False):
      Similarity in [0.363, 0.440) or margin < 0.035 or degraded quality.
  - NO_RELIABLE_MATCH (Tier: NO_MATCH, is_match=False):
      Similarity < 0.363 or no face detected.
"""
from __future__ import annotations

from app.face.identity_decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    MIN_MARGIN_FOR_HIGH_MATCH,
    evaluate_identity_decision,
)
from app.models.schemas import IdentityDecision

# Aliases for explicit architectural terminology requested in specs
HIGH_CONFIDENCE_MATCH = "HIGH CONFIDENCE MATCH"
POSSIBLE_MATCH_REVIEW = "REVIEW / UNCERTAIN"
NO_RELIABLE_MATCH = "NO MATCH"

__all__ = [
    "BASELINE_MATCH_THRESHOLD",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MIN_MARGIN_FOR_HIGH_MATCH",
    "HIGH_CONFIDENCE_MATCH",
    "POSSIBLE_MATCH_REVIEW",
    "NO_RELIABLE_MATCH",
    "IdentityDecision",
    "evaluate_identity_decision",
]
