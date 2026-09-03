"""
app/identity — Forensic Identity Decision & Calibration Package.
"""
from __future__ import annotations

from app.identity.calibration import (
    calibrate_self_match,
    evaluate_synthetic_benchmark,
    run_target_augmentation_robustness,
)
from app.identity.decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_MATCH,
    HIGH_CONFIDENCE_THRESHOLD,
    MIN_MARGIN_FOR_HIGH_MATCH,
    NO_RELIABLE_MATCH,
    POSSIBLE_MATCH_REVIEW,
    IdentityDecision,
    evaluate_identity_decision,
)

__all__ = [
    "BASELINE_MATCH_THRESHOLD",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MIN_MARGIN_FOR_HIGH_MATCH",
    "HIGH_CONFIDENCE_MATCH",
    "POSSIBLE_MATCH_REVIEW",
    "NO_RELIABLE_MATCH",
    "IdentityDecision",
    "evaluate_identity_decision",
    "calibrate_self_match",
    "run_target_augmentation_robustness",
    "evaluate_synthetic_benchmark",
]
