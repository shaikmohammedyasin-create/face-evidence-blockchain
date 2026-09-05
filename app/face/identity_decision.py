"""
app/face/identity_decision.py — Forensic Identity Decision Engine.

Separates raw machine-learning cosine similarity values from application-level
forensic identity determinations.

Applies a three-tier decision state machine:
  1. HIGH CONFIDENCE MATCH (Tier: HIGH_MATCH, is_match=True):
     - Cosine similarity >= HIGH_CONFIDENCE_THRESHOLD (0.45)
     - Distinct margin over runner-up candidates from different sources (>= 0.035)
     - High facial image quality (resolution, sharpness, lighting)
     - Unambiguous landmark alignment

  2. REVIEW / UNCERTAIN (Tier: REVIEW, is_match=False):
     - Cosine similarity meets basic baseline (>= 0.363) but falls below high confidence
     - OR candidate score is clustered near runner-ups (margin < 0.035, ambiguity detected)
     - OR candidate thumbnail has degraded resolution / blur / poor illumination
     - Requires human operator inspection before drawing conclusions

  3. NO MATCH (Tier: NO_MATCH, is_match=False):
     - Cosine similarity < 0.363 (official SFace identity boundary)
     - Or no faces detected in candidate image
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.config import FACE_MATCH_THRESHOLD
from app.face.quality import FaceQualityAssessment
from app.models.schemas import IdentityDecision, MatchResult
from app.utils.logger import get_logger

log = get_logger(__name__)

# Calibrated Threshold Boundaries
BASELINE_MATCH_THRESHOLD: float = FACE_MATCH_THRESHOLD  # 0.363
HIGH_CONFIDENCE_THRESHOLD: float = 0.440
MIN_MARGIN_FOR_HIGH_MATCH: float = 0.035
DEFAULT_MIN_Z_SCORE: float = 2.0


def evaluate_identity_decision(
    best_similarity: float,
    runner_up_similarity: float,
    quality: Optional[FaceQualityAssessment] = None,
    matched_face_index: int = 0,
    candidate_faces_count: int = 1,
    candidate_domain: str = "",
    runner_up_domain: str = "",
    candidate_pool_scores: Optional[list[float]] = None,
    min_z_score: float = DEFAULT_MIN_Z_SCORE,
) -> IdentityDecision:
    """
    Evaluate forensic identity confidence tier for the top candidate.

    Args:
        best_similarity: Cosine similarity of the top candidate face (-1.0 to 1.0).
        runner_up_similarity: Cosine similarity of the highest runner-up candidate NOT in top cluster.
        quality: Forensic quality evaluation of the candidate face crop.
        matched_face_index: Index of matched face in candidate image.
        candidate_faces_count: Total faces detected in candidate image.
        candidate_domain: Web domain of candidate.
        runner_up_domain: Web domain of runner-up candidate.
        candidate_pool_scores: List of similarity scores across all search candidates.
        min_z_score: Minimum Z-score above candidate pool mean required for HIGH_MATCH (default: 2.0).

    Returns:
        Structured :class:`IdentityDecision` object.
    """
    import numpy as np

    margin = max(0.0, best_similarity - runner_up_similarity) if runner_up_similarity > 0 else best_similarity
    quality_passed = quality.is_acceptable if quality else True
    quality_score = quality.score if quality else 1.0
    quality_details = quality.to_dict() if quality else {}

    # Calculate per-query statistical pool metrics (Z-score outlier test)
    if candidate_pool_scores and len(candidate_pool_scores) > 1:
        # Collapse near-duplicate scores (difference < 0.005) so duplicate hits do not inflate background pool mean
        sorted_pool = sorted([float(s) for s in candidate_pool_scores if s is not None], reverse=True)
        unique_pool: list[float] = []
        for s in sorted_pool:
            if not any(abs(s - u) < 0.005 for u in unique_pool):
                unique_pool.append(s)

        pool_mean = float(np.mean(unique_pool))
        pool_std = float(np.std(unique_pool))
        if pool_std > 1e-6:
            z_score = float((best_similarity - pool_mean) / pool_std)
        else:
            z_score = 3.0 if best_similarity >= HIGH_CONFIDENCE_THRESHOLD else 0.0

        # Adjust required Z-score for small sample sizes N where Samuelson's inequality bounds max Z to sqrt(N-1)
        max_possible_z = float(np.sqrt(len(unique_pool) - 1)) if len(unique_pool) > 1 else 3.0
        effective_min_z = min(min_z_score, max(1.2, 0.85 * max_possible_z))
    else:
        pool_mean = float(best_similarity)
        pool_std = 0.0
        z_score = 3.0 if best_similarity >= HIGH_CONFIDENCE_THRESHOLD else 0.0
        effective_min_z = min_z_score

    is_statistical_outlier = z_score >= effective_min_z

    warnings: list[str] = []
    if quality and quality.reasons:
        warnings.extend(quality.reasons)

    if candidate_faces_count > 1:
        warnings.append(
            f"Multi-face image ({candidate_faces_count} faces detected) — matched face #{matched_face_index + 1}"
        )

    # 1. Below baseline threshold -> Definite NO MATCH
    if best_similarity < BASELINE_MATCH_THRESHOLD:
        reason = (
            f"Cosine similarity ({best_similarity:.3f}) is below the biometric threshold "
            f"({BASELINE_MATCH_THRESHOLD:.3f}). No statistical correlation."
        )
        return IdentityDecision(
            verdict="NO MATCH",
            tier="NO_MATCH",
            is_match=False,
            confidence=best_similarity,
            margin=margin,
            runner_up_confidence=runner_up_similarity,
            matched_face_index=matched_face_index,
            candidate_faces_count=candidate_faces_count,
            quality_passed=quality_passed,
            quality_score=quality_score,
            quality_details=quality_details,
            decision_reason=reason,
            warnings=warnings,
            match_factors={
                "similarity_level": "Sub-threshold",
                "margin_status": "N/A",
                "quality_status": "N/A",
                "statistical_status": f"Sub-threshold (μ={pool_mean:.3f}, σ={pool_std:.3f})",
            },
            z_score=round(z_score, 2),
            pool_mean=round(pool_mean, 4),
            pool_std=round(pool_std, 4),
            is_statistical_outlier=is_statistical_outlier,
        )

    # 2. Above baseline threshold — Determine High Match vs Review
    is_high_sim = best_similarity >= HIGH_CONFIDENCE_THRESHOLD
    has_good_margin = margin >= MIN_MARGIN_FOR_HIGH_MATCH or runner_up_similarity == 0.0

    if not has_good_margin and runner_up_similarity >= BASELINE_MATCH_THRESHOLD:
        warnings.append(
            f"Score clustering: non-cluster runner-up similarity ({runner_up_similarity:.3f}) is within "
            f"margin delta ({margin:.3f} < {MIN_MARGIN_FOR_HIGH_MATCH:.3f}) of top candidate"
        )

    if not is_statistical_outlier and candidate_pool_scores and len(candidate_pool_scores) > 1:
        warnings.append(
            f"No statistical separation: Z-score ({z_score:.2f}) is below relative threshold ({min_z_score:.2f}) "
            f"above pool mean ({pool_mean:.3f})"
        )

    if is_high_sim and has_good_margin and quality_passed and is_statistical_outlier:
        verdict = "HIGH CONFIDENCE MATCH"
        tier = "HIGH_MATCH"
        is_match = True
        reason = (
            f"High-confidence biometric match: Cosine similarity ({best_similarity:.3f}) "
            f"comfortably exceeds high threshold ({HIGH_CONFIDENCE_THRESHOLD:.3f}) with "
            f"+{margin:.3f} margin over non-duplicate candidates and statistical separation (Z={z_score:.2f} >= {min_z_score:.1f})."
        )
        factors = {
            "similarity_level": f"Strong ({best_similarity:.3f} >= {HIGH_CONFIDENCE_THRESHOLD:.3f})",
            "margin_status": f"Clear separation (+{margin:.3f})",
            "quality_status": f"Passed ({quality_details.get('quality_label', 'Good')})",
            "statistical_status": f"Outlier (Z={z_score:.2f} >= {min_z_score:.1f}, μ={pool_mean:.3f}, σ={pool_std:.3f})",
        }
    else:
        verdict = "REVIEW / UNCERTAIN"
        tier = "REVIEW"
        is_match = False  # Review tier never automatically declares a verified match

        reasons_list: list[str] = []
        if not is_high_sim:
            reasons_list.append(
                f"Similarity ({best_similarity:.3f}) is in moderate confidence zone "
                f"({BASELINE_MATCH_THRESHOLD:.3f} – {HIGH_CONFIDENCE_THRESHOLD:.3f})"
            )
        if not has_good_margin and runner_up_similarity > 0:
            reasons_list.append(
                f"Low separation from non-duplicate runner-up ({runner_up_similarity:.3f}, margin: {margin:.3f})"
            )
        if not is_statistical_outlier:
            reasons_list.append(
                f"Lack of statistical separation from candidate pool (Z={z_score:.2f} < {min_z_score:.1f})"
            )
        if not quality_passed:
            reasons_list.append("Face crop did not meet forensic quality standards")

        reason = "Manual Review Advised: " + "; ".join(reasons_list) + "."
        factors = {
            "similarity_level": f"Moderate ({best_similarity:.3f})",
            "margin_status": f"Cluster margin (+{margin:.3f})",
            "quality_status": quality_details.get("quality_label", "Degraded") if not quality_passed else "Acceptable",
            "statistical_status": f"Not Outlier (Z={z_score:.2f} < {min_z_score:.1f}, μ={pool_mean:.3f}, σ={pool_std:.3f})",
        }

    return IdentityDecision(
        verdict=verdict,
        tier=tier,
        is_match=is_match,
        confidence=best_similarity,
        margin=margin,
        runner_up_confidence=runner_up_similarity,
        matched_face_index=matched_face_index,
        candidate_faces_count=candidate_faces_count,
        quality_passed=quality_passed,
        quality_score=quality_score,
        quality_details=quality_details,
        decision_reason=reason,
        warnings=warnings,
        match_factors=factors,
        z_score=round(z_score, 2),
        pool_mean=round(pool_mean, 4),
        pool_std=round(pool_std, 4),
        is_statistical_outlier=is_statistical_outlier,
    )
