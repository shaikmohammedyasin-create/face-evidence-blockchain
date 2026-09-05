"""
tests/test_identity_decision.py — Tests for the Forensic Identity Decision Engine.
"""
from __future__ import annotations

import pytest

from app.face.identity_decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    evaluate_identity_decision,
)
from app.face.quality import FaceQualityAssessment


def _mock_quality(acceptable: bool = True, score: float = 0.9) -> FaceQualityAssessment:
    return FaceQualityAssessment(
        is_acceptable=acceptable,
        score=score,
        width=100,
        height=100,
        laplacian_variance=80.0 if acceptable else 15.0,
        mean_brightness=120.0,
        contrast_std=35.0,
        interocular_dist_ratio=0.4,
        detection_confidence=0.98,
        reasons=[] if acceptable else ["Blurry crop"],
        quality_label="Good" if acceptable else "Degraded",
    )


def test_high_confidence_match():
    dec = evaluate_identity_decision(
        best_similarity=0.55,
        runner_up_similarity=0.30,
        quality=_mock_quality(acceptable=True, score=0.95),
        matched_face_index=0,
        candidate_faces_count=1,
    )
    assert dec.tier == "HIGH_MATCH"
    assert dec.verdict == "HIGH CONFIDENCE MATCH"
    assert dec.is_match is True
    assert dec.margin >= 0.20
    assert "High-confidence biometric match" in dec.decision_reason


def test_sub_threshold_no_match():
    dec = evaluate_identity_decision(
        best_similarity=0.25,
        runner_up_similarity=0.20,
        quality=_mock_quality(acceptable=True),
    )
    assert dec.tier == "NO_MATCH"
    assert dec.verdict == "NO MATCH"
    assert dec.is_match is False
    assert "below the biometric threshold" in dec.decision_reason


def test_review_moderate_similarity():
    # Between 0.363 and 0.440
    dec = evaluate_identity_decision(
        best_similarity=0.395,
        runner_up_similarity=0.20,
        quality=_mock_quality(acceptable=True),
    )
    assert dec.tier == "REVIEW"
    assert dec.verdict == "REVIEW / UNCERTAIN"
    assert dec.is_match is False
    assert "moderate confidence zone" in dec.decision_reason


def test_review_low_margin_ambiguity():
    # High similarity (0.50) but runner-up is very close (0.49), indicating ambiguous cluster
    dec = evaluate_identity_decision(
        best_similarity=0.50,
        runner_up_similarity=0.49,
        quality=_mock_quality(acceptable=True),
    )
    assert dec.tier == "REVIEW"
    assert dec.verdict == "REVIEW / UNCERTAIN"
    assert dec.is_match is False
    assert any("clustering" in w.lower() or "separation" in dec.decision_reason.lower() for w in dec.warnings + [dec.decision_reason])


def test_review_poor_quality():
    # High similarity (0.55) but degraded face quality
    dec = evaluate_identity_decision(
        best_similarity=0.55,
        runner_up_similarity=0.20,
        quality=_mock_quality(acceptable=False, score=0.3),
    )
    assert dec.tier == "REVIEW"
    assert dec.verdict == "REVIEW / UNCERTAIN"
    assert dec.is_match is False
    assert "quality" in dec.decision_reason.lower()


def test_multi_face_warning():
    dec = evaluate_identity_decision(
        best_similarity=0.52,
        runner_up_similarity=0.25,
        quality=_mock_quality(acceptable=True),
        matched_face_index=1,
        candidate_faces_count=3,
    )
    assert any("matched face #2" in w.lower() for w in dec.warnings)


def test_near_duplicate_runner_up_high_confidence():
    """
    Test case (a): True match (0.9563) with a near-duplicate runner-up (0.9548)
    where non-cluster runner-up is 0.4693. Should classify as HIGH CONFIDENCE MATCH.
    """
    pool_scores = [0.9563, 0.9548, 0.4693, 0.450, 0.440, 0.430, 0.420, 0.410, 0.400, 0.390]
    dec = evaluate_identity_decision(
        best_similarity=0.9563,
        runner_up_similarity=0.4693,  # Non-cluster runner-up score
        quality=_mock_quality(acceptable=True, score=0.95),
        candidate_pool_scores=pool_scores,
    )
    assert dec.tier == "HIGH_MATCH"
    assert dec.verdict == "HIGH CONFIDENCE MATCH"
    assert dec.is_match is True
    assert dec.margin > 0.45
    assert dec.is_statistical_outlier is True
    assert dec.z_score >= 2.0


def test_unrelated_domain_moderate_similarity_no_statistical_separation():
    """
    Test case (b): Moderate similarity candidate (0.4693) from unrelated domain
    lacking statistical separation from the background candidate pool (Z-score < 2.0).
    Should classify as REVIEW / UNCERTAIN, not HIGH MATCH.
    """
    pool_scores = [0.4693, 0.4650, 0.4610, 0.4590, 0.4550, 0.4500, 0.4480, 0.4420]
    dec = evaluate_identity_decision(
        best_similarity=0.4693,
        runner_up_similarity=0.4650,
        quality=_mock_quality(acceptable=True),
        candidate_pool_scores=pool_scores,
    )
    assert dec.tier == "REVIEW"
    assert dec.verdict == "REVIEW / UNCERTAIN"
    assert dec.is_match is False
    assert dec.is_statistical_outlier is False
    assert dec.z_score < 2.0
    assert any("statistical" in w.lower() or "separation" in w.lower() for w in dec.warnings)
