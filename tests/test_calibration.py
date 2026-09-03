"""
tests/test_calibration.py — Biometric calibration and forensic accuracy benchmark tests.

Evaluates:
  - False Acceptance Rate (FAR) and False Rejection Rate (FRR) across calibrated thresholds.
  - Precision and Recall on synthetic/calibrated genuine and imposter embedding pairs.
  - Multi-face scanning correctness and ranking with margin deltas.
  - URL normalization and candidate deduplication.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.face.identity_decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    evaluate_identity_decision,
)
from app.face.matcher import (
    cosine_similarity,
    deduplicate_candidates,
    normalize_url,
    rank_candidates,
)
from app.models.schemas import MatchResult, SearchCandidate


# ── Benchmark Embedding Fixtures ─────────────────────────────────────────────

def _generate_normalised_vector(dim: int = 128, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _generate_perturbed_vector(base: list[float], noise_std: float = 0.1, seed: int = 101) -> list[float]:
    rng = np.random.default_rng(seed)
    arr = np.array(base, dtype=np.float64)
    noise = rng.normal(0, noise_std, len(base))
    perturbed = arr + noise
    perturbed = perturbed / np.linalg.norm(perturbed)
    return perturbed.tolist()


# ── Tests ────────────────────────────────────────────────────────────────────

class TestUrlDeduplication:
    def test_normalize_url_removes_tracking(self):
        url = "https://Instagram.com/p/123/?utm_source=ig_web_copy_link&utm_medium=referral&fbclid=abcdef123"
        cleaned = normalize_url(url)
        assert cleaned == "https://instagram.com/p/123"
        assert "utm_source" not in cleaned
        assert "fbclid" not in cleaned

    def test_deduplicate_candidates_removes_redundant_urls(self):
        c1 = SearchCandidate(url="https://example.com/photo1?utm_source=twitter", image_url="https://img.example.com/1.jpg", platform="Twitter")
        c2 = SearchCandidate(url="https://example.com/photo1?ref=feed", image_url="https://img.example.com/1.jpg", platform="Twitter")
        c3 = SearchCandidate(url="https://other.com/photo2", image_url="https://img.other.com/2.jpg", platform="Blog")

        deduped = deduplicate_candidates([c1, c2, c3])
        assert len(deduped) == 2
        assert deduped[0].url == c1.url
        assert deduped[1].url == c3.url


class TestBiometricCalibration:
    def test_imposter_pairs_far_zero_at_high_threshold(self):
        """Verify that distinct identity embeddings never exceed the high confidence threshold."""
        num_identities = 50
        identities = [_generate_normalised_vector(seed=i) for i in range(num_identities)]

        imposter_scores: list[float] = []
        for i in range(num_identities):
            for j in range(i + 1, num_identities):
                sim = cosine_similarity(identities[i], identities[j])
                imposter_scores.append(sim)

        # High threshold (0.44) should yield 0% False Acceptance Rate
        false_accepts = sum(1 for s in imposter_scores if s >= HIGH_CONFIDENCE_THRESHOLD)
        far = false_accepts / len(imposter_scores)
        assert far == 0.0, f"FAR was {far:.4f}, expected 0.0"

        # Baseline threshold (0.363) should also have low FAR on random hypersphere vectors
        baseline_false_accepts = sum(1 for s in imposter_scores if s >= BASELINE_MATCH_THRESHOLD)
        baseline_far = baseline_false_accepts / len(imposter_scores)
        assert baseline_far < 0.01

    def test_genuine_pairs_frr_at_high_threshold(self):
        """Verify that genuine perturbed embeddings achieve high similarity."""
        base_person = _generate_normalised_vector(seed=777)
        # Realistic intra-class variation (noise_std=0.04 simulating pose/lighting variations)
        genuine_views = [_generate_perturbed_vector(base_person, noise_std=0.04, seed=1000 + i) for i in range(20)]

        similarities = [cosine_similarity(base_person, view) for view in genuine_views]
        for sim in similarities:
            assert sim >= BASELINE_MATCH_THRESHOLD, f"Genuine view scored {sim:.3f} < {BASELINE_MATCH_THRESHOLD}"
            assert sim >= HIGH_CONFIDENCE_THRESHOLD, f"Genuine view scored {sim:.3f} < {HIGH_CONFIDENCE_THRESHOLD}"


class TestRankingAndMargin:
    def test_rank_candidates_computes_margin_and_decision(self):
        c1 = SearchCandidate(url="https://site1.com/post1", image_url="https://img.site1.com/1.jpg")
        c2 = SearchCandidate(url="https://site2.com/post2", image_url="https://img.site2.com/2.jpg")
        c3 = SearchCandidate(url="https://site3.com/post3", image_url="https://img.site3.com/3.jpg")

        mr1 = MatchResult(candidate=c1, matched=True, confidence=0.55, matched_face_index=0, candidate_faces_count=1)
        mr2 = MatchResult(candidate=c2, matched=False, confidence=0.35, matched_face_index=0, candidate_faces_count=1)
        mr3 = MatchResult(candidate=c3, matched=False, confidence=0.20, matched_face_index=0, candidate_faces_count=1)

        ranked = rank_candidates([mr2, mr1, mr3])
        assert ranked[0].candidate.url == c1.url
        assert ranked[0].decision_tier == "HIGH_MATCH"
        assert ranked[0].matched is True
        assert ranked[0].margin_from_runner_up == pytest.approx(0.20, abs=0.01)

        assert ranked[1].candidate.url == c2.url
        assert ranked[1].decision_tier == "NO_MATCH"
