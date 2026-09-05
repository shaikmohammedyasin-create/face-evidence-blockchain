"""
tests/test_bugs_abc.py — Unit & Integration tests for Bug A, Bug B, Bug C,
Verification Certificate Generation, and Negative-Case Fixture.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from app.demo import run_demo
from app.evidence.certificate import generate_verification_certificate
from app.face.matcher import rank_candidates
from app.models.schemas import MatchResult, SearchCandidate, SearchCandidateList
from app.search.result_validator import classify_url_source, is_search_result_url


class TestBugAMarginCalculation:
    """BUG A Test — Identity Margin Mathematical Consistency."""

    def test_margin_subtraction_close_scores(self):
        c1 = SearchCandidate(url="https://example.com/c1", platform="Web")
        c2 = SearchCandidate(url="https://example.com/c2", platform="Web")

        m1 = MatchResult(candidate=c1, matched=True, confidence=0.956)
        m2 = MatchResult(candidate=c2, matched=True, confidence=0.955)

        ranked = rank_candidates([m1, m2])
        top = ranked[0]

        # Raw margin = 0.956 - 0.955 = 0.001
        assert pytest.approx(top.margin_from_runner_up, abs=1e-4) == 0.001
        assert pytest.approx(top.runner_up_similarity, abs=1e-4) == 0.955

        # Displayed percentage calculation: 95.6 - 95.5 = 0.1
        top_pct = round(top.confidence * 100, 1)
        second_pct = round(top.runner_up_similarity * 100, 1)
        displayed_margin = round(top_pct - second_pct, 1)
        assert displayed_margin == 0.1

    def test_margin_subtraction_wide_scores(self):
        c1 = SearchCandidate(url="https://example.com/c1", platform="Web")
        c2 = SearchCandidate(url="https://example.com/c2", platform="Web")

        m1 = MatchResult(candidate=c1, matched=True, confidence=0.800)
        m2 = MatchResult(candidate=c2, matched=False, confidence=0.650)

        ranked = rank_candidates([m1, m2])
        top = ranked[0]

        top_pct = round(top.confidence * 100, 1)
        second_pct = round(top.runner_up_similarity * 100, 1)
        displayed_margin = round(top_pct - second_pct, 1)
        assert displayed_margin == 15.0


class TestBugBSearchSummaryCounts:
    """BUG B Test — Authoritative Pipeline Search Statistics Single Source of Truth."""

    def test_search_stats_consistency(self, monkeypatch, tmp_path):
        from app.pipeline.orchestrator import run_pipeline

        test_img = tmp_path / "test.jpg"
        test_img.write_bytes(b"dummy image data")

        cand1 = SearchCandidate(url="https://example.com/1", image_url="https://example.com/1.jpg", platform="Web")
        cand2 = SearchCandidate(url="https://example.com/2", image_url="https://example.com/2.jpg", platform="Web")

        # Mock heavy stages for unit speed
        monkeypatch.setattr("app.pipeline.orchestrator.validate_image_path", lambda p: Path(p))
        monkeypatch.setattr("app.pipeline.orchestrator.sha256_file", lambda p: "fakehash")
        monkeypatch.setattr("app.pipeline.orchestrator.assess_image_quality", lambda p: {"resolution": "100x100", "summary": "Good"})
        monkeypatch.setattr("app.pipeline.orchestrator.step_detect_faces", lambda p, face_index=None: ([{"facial_area": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.99}], 0))
        monkeypatch.setattr("app.pipeline.orchestrator.step_generate_embedding", lambda p, idx: type("FaceResult", (), {"embedding": [0.1]*128, "embedding_generated": True, "embedding_dimension": 128})())
        monkeypatch.setattr("app.pipeline.orchestrator.calibrate_self_match", lambda emb: 1.0)
        def _mock_search(p, max_candidates=30):
            return SearchCandidateList(
                [cand1, cand2],
                raw_candidates_count=208,
                unique_candidates_count=30,
                usable_candidates_count=2,
            )

        monkeypatch.setattr("app.pipeline.orchestrator.step_search", _mock_search)
        monkeypatch.setattr("app.pipeline.orchestrator.step_match_candidates", lambda face_res, cands, progress_cb=None: [MatchResult(candidate=c, matched=True, confidence=0.95, candidate_faces_count=1) for c in cands])
        monkeypatch.setattr("app.pipeline.orchestrator.step_fingerprint", lambda match, p, sha: (type("EvidenceRecord", (), {})(), type("Fingerprint", (), {"fingerprint": "abc12345"})()))
        monkeypatch.setattr("app.pipeline.orchestrator.step_upload", lambda fp, blockchain_network=None: (type("BlockchainRecord", (), {"network": "Simulated", "transaction_hash": "0x123", "status": "simulated"})(), None))
        monkeypatch.setattr("app.pipeline.orchestrator.step_verify", lambda rec, ev, client: type("VerificationResult", (), {"verified": True, "current_fingerprint": "abc12345", "stored_fingerprint": "abc12345"})())
        monkeypatch.setattr("app.pipeline.orchestrator.demonstrate_tamper", lambda ev: {"tamper_detected": True, "original_hash": "a", "tampered_hash": "b"})

        res = run_pipeline(test_img, quiet=True)
        stats = res.get("search_stats", {})

        assert stats["raw_candidates_count"] == 208
        assert stats["unique_candidates_count"] == 30
        assert stats["usable_candidates_count"] == 2
        assert stats["face_usable_candidates_count"] == 2


class TestBugCUrlClassification:
    """BUG C Test — Search Result Page vs Direct Profile URL Classification."""

    def test_is_search_result_url_detection(self):
        assert is_search_result_url("https://www.linkedin.com/pub/dir/+/Mohammed+Yasin") is True
        assert is_search_result_url("https://www.linkedin.com/dir/+/Mohammed+Yasin") is True
        assert is_search_result_url("https://example.com/search?q=test") is True
        assert is_search_result_url("https://example.com/results?id=1") is True

        assert is_search_result_url("https://www.linkedin.com/in/mohammed-yasin") is False
        assert is_search_result_url("https://www.instagram.com/p/C123456/") is False
        assert is_search_result_url("https://x.com/user/status/123456") is False

    def test_classify_url_source_label(self):
        assert classify_url_source("https://www.linkedin.com/pub/dir/+/Mohammed+Yasin") == "SEARCH RESULT PAGE — NOT A DIRECT PROFILE"
        assert classify_url_source("https://www.linkedin.com/in/mohammed-yasin") == "DIRECT POST / PROFILE"


class TestVerificationCertificateArtifact:
    """Task 1 Test — Verification Certificate HTML Generation."""

    def test_certificate_file_creation_and_contents(self, tmp_path):
        cert_path = generate_verification_certificate(
            input_image_sha256="50704997c42fe612c840770d34507c47ae147e035f663a407badaa22897ce633",
            matched_candidate_url="https://www.linkedin.com/pub/dir/+/Mohammed+Yasin",
            matched_platform="LinkedIn",
            decision_tier="HIGH_MATCH",
            cosine_similarity=0.9563,
            margin=0.4900,
            transaction_hash="0x5b90cf254c0f706539d56de5fc1c188247c3e77ede6cec7551b82ed7a8ed2407",
            contract_address="0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
            network_name="Local EVM Simulation",
            output_dir=tmp_path,
        )

        assert cert_path.is_file()
        assert cert_path.name.startswith("verification_certificate_")
        assert cert_path.name.endswith(".html")

        content = cert_path.read_text(encoding="utf-8")
        assert "50704997c42fe612c840770d34507c47ae147e035f663a407badaa22897ce633" in content
        assert "https://www.linkedin.com/pub/dir/+/Mohammed+Yasin" in content
        assert "LinkedIn" in content
        assert "95.6%" in content
        assert "0x5b90cf254c0f706539d56de5fc1c188247c3e77ede6cec7551b82ed7a8ed2407" in content
        assert "0x71C7656EC7ab88b098defB751B7401B5f6d8976F" in content
        assert "data:image/png;base64," in content  # Embedded QR Code


class TestNegativeCaseFixture:
    """Task 2 Test — Negative-Case Fixture Demonstrates NO RELIABLE MATCH FOUND."""

    def test_negative_case_returns_no_match(self, monkeypatch):
        monkeypatch.setattr("app.search.reverse_search.search_for_image", lambda img, preferred_provider=None, search_multi_representations=False: [SearchCandidate(url="https://example.com/neg", image_url="https://example.com/neg.jpg")])
        ret = run_demo(demo_negative_case=True)
        assert ret == 0
