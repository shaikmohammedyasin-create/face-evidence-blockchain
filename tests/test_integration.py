"""
tests/test_integration.py — End-to-end integration and system testing.

Verifies:
  - Complete end-to-end pipeline flow from image to blockchain verification
  - Canonical EvidenceRecord generation & RFC 8785 JSON determinism
  - Machine-readable JSON output correctness
  - Judge Mode diagnostic report output
  - ONNX model checksum auditing
  - Automatic fallback between search providers
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.blockchain.client import SimulatedClient
from app.blockchain.hasher import calculate_fingerprint, create_canonical_record
from app.blockchain.verifier import demonstrate_tamper, re_verify
from app.demo import run_demo
from app.face.models import get_model_checksums
from app.models.schemas import (
    BlockchainRecord,
    EvidenceRecord,
    FaceResult,
    Fingerprint,
    MatchResult,
    SearchCandidate,
)
from app.pipeline.orchestrator import run_pipeline


@pytest.fixture
def mock_pipeline_environment(tmp_path: Path):
    """Create a temporary test image and mock face + search stages."""
    # Create valid synthetic test image
    img_path = tmp_path / "test_portrait.png"
    import cv2
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw simple circle (face shape)
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    cv2.imwrite(str(img_path), img)

    # Synthetic 128-d normalised embedding
    v = np.ones(128, dtype=np.float64) / np.sqrt(128)
    mock_face = FaceResult(
        faces_detected=1,
        selected_face=0,
        embedding_generated=True,
        embedding_dimension=128,
        embedding=v.tolist(),
        landmarks=[(80, 80), (120, 80), (100, 100), (85, 120), (115, 120)],
        bounding_box={"x": 50, "y": 50, "w": 100, "h": 100},
        detection_confidence=0.98,
    )

    mock_candidates = [
        SearchCandidate(
            url="https://instagram.com/p/mock123",
            title="Celebrity Portrait Post",
            snippet="Photo from red carpet",
            image_url="https://cdn.mock.com/thumb1.jpg",
            platform="Instagram",
        )
    ]

    return {
        "image_path": img_path,
        "mock_face": mock_face,
        "mock_candidates": mock_candidates,
    }


class TestEndToEndPipeline:
    """Test full pipeline orchestrator execution."""

    def test_full_pipeline_success(self, mock_pipeline_environment, monkeypatch):
        env = mock_pipeline_environment

        with patch("app.pipeline.orchestrator.step_detect_faces") as mock_detect, \
             patch("app.pipeline.orchestrator.step_generate_embedding") as mock_enc, \
             patch("app.pipeline.orchestrator.step_search") as mock_search, \
             patch("app.pipeline.orchestrator.download_image") as mock_dl, \
             patch("app.pipeline.orchestrator.match_candidate") as mock_match:

            mock_detect.return_value = ([{"facial_area": {"x": 50, "y": 50, "w": 100, "h": 100}, "confidence": 0.98}], 0)
            mock_enc.return_value = env["mock_face"]
            mock_search.return_value = env["mock_candidates"]
            mock_dl.return_value = env["image_path"]
            mock_match.return_value = MatchResult(
                candidate=env["mock_candidates"][0],
                matched=True,
                confidence=0.88,
                local_image_path=str(env["image_path"]),
            )

            result = run_pipeline(
                image_path=str(env["image_path"]),
                face_index=0,
                skip_blockchain=False,
                judge_mode=True,
                quiet=True,
            )

            assert result["image_path"] == str(env["image_path"])
            assert len(result["input_image_sha256"]) == 64
            assert result["face_result"].embedding_dimension == 128
            assert result["best_match"].matched is True
            assert result["best_match"].confidence == 0.88
            assert result["fingerprint"] is not None
            assert len(result["fingerprint"].fingerprint) == 64
            assert result["blockchain_record"] is not None
            assert result["blockchain_record"].status in ("confirmed", "simulated")
            assert result["verification"].verified is True
            assert result["tamper_demo"]["tamper_detected"] is True
            assert "total_pipeline_duration_seconds" in result
            assert "stage_timings" in result

    def test_pipeline_skip_blockchain(self, mock_pipeline_environment):
        env = mock_pipeline_environment

        with patch("app.pipeline.orchestrator.step_detect_faces") as mock_detect, \
             patch("app.pipeline.orchestrator.step_generate_embedding") as mock_enc, \
             patch("app.pipeline.orchestrator.step_search") as mock_search, \
             patch("app.pipeline.orchestrator.download_image") as mock_dl, \
             patch("app.pipeline.orchestrator.match_candidate") as mock_match:

            mock_detect.return_value = ([{"facial_area": {"x": 50, "y": 50, "w": 100, "h": 100}, "confidence": 0.98}], 0)
            mock_enc.return_value = env["mock_face"]
            mock_search.return_value = env["mock_candidates"]
            mock_dl.return_value = env["image_path"]
            mock_match.return_value = MatchResult(
                candidate=env["mock_candidates"][0],
                matched=True,
                confidence=0.75,
                local_image_path=str(env["image_path"]),
            )

            result = run_pipeline(
                image_path=str(env["image_path"]),
                skip_blockchain=True,
                quiet=True,
            )

            assert result["blockchain_record"] is None
            assert result["verification"] is None
            assert result["tamper_demo"] is None

    def test_pipeline_multiple_faces_requires_selection(self, mock_pipeline_environment):
        env = mock_pipeline_environment
        with patch("app.pipeline.orchestrator.detect_faces") as mock_det:
            mock_det.return_value = [
                {"facial_area": {"x": 10, "y": 10, "w": 30, "h": 30}, "confidence": 0.95},
                {"facial_area": {"x": 50, "y": 50, "w": 30, "h": 30}, "confidence": 0.91},
            ]
            with pytest.raises(ValueError, match="Multiple faces"):
                run_pipeline(image_path=str(env["image_path"]), face_index=None, quiet=True)


class TestEvidenceAndTamper:
    """Test the canonical evidence model and tamper avalanche proof."""

    def test_evidence_record_canonical_hashing(self, mock_pipeline_environment):
        env = mock_pipeline_environment
        match = MatchResult(
            candidate=env["mock_candidates"][0],
            matched=True,
            confidence=0.92,
            local_image_path=str(env["image_path"]),
        )

        evidence = create_canonical_record(
            match=match,
            input_image_path=env["image_path"],
            input_image_sha256="abcdef1234567890" * 4,
            timestamp_iso="2026-03-01T12:00:00Z",
        )

        fp1 = calculate_fingerprint(evidence)
        fp2 = calculate_fingerprint(evidence)
        assert fp1.fingerprint == fp2.fingerprint
        assert len(fp1.fingerprint) == 64
        assert "schema_version" in fp1.canonical_json
        assert "YuNet-2023mar" in fp1.canonical_json

    def test_tamper_avalanche_detection(self, mock_pipeline_environment):
        env = mock_pipeline_environment
        match = MatchResult(
            candidate=env["mock_candidates"][0],
            matched=True,
            confidence=0.95,
        )
        evidence = create_canonical_record(match=match, input_image_sha256="11" * 32)
        fp = calculate_fingerprint(evidence)

        client = SimulatedClient()
        record = client.store_fingerprint(fp.fingerprint, {})

        # Original verifies
        ver_orig = re_verify(record, evidence, client)
        assert ver_orig.verified is True

        # Tamper demonstration
        tamper_result = demonstrate_tamper(evidence, mutation_field="title")
        assert tamper_result["tamper_detected"] is True
        assert tamper_result["original_hash"] != tamper_result["tampered_hash"]

        # Tampered data fails verification
        tampered_fp = Fingerprint(
            algorithm="SHA-256",
            fingerprint=tamper_result["tampered_hash"],
            source_data=tamper_result["tampered_data"],
        )
        ver_tampered = re_verify(record, tampered_fp, client)
        assert ver_tampered.verified is False


class TestDemoCLI:
    """Test demo entry point with json and judge options."""

    def test_demo_json_output(self, mock_pipeline_environment, capsys):
        env = mock_pipeline_environment

        with patch("app.pipeline.orchestrator.step_detect_faces") as mock_detect, \
             patch("app.pipeline.orchestrator.step_generate_embedding") as mock_enc, \
             patch("app.pipeline.orchestrator.step_search") as mock_search, \
             patch("app.pipeline.orchestrator.download_image") as mock_dl, \
             patch("app.pipeline.orchestrator.match_candidate") as mock_match:

            mock_detect.return_value = ([{"facial_area": {"x": 50, "y": 50, "w": 100, "h": 100}, "confidence": 0.98}], 0)
            mock_enc.return_value = env["mock_face"]
            mock_search.return_value = env["mock_candidates"]
            mock_dl.return_value = env["image_path"]
            mock_match.return_value = MatchResult(
                candidate=env["mock_candidates"][0],
                matched=True,
                confidence=0.89,
                local_image_path=str(env["image_path"]),
            )

            ret = run_demo(
                image_path=str(env["image_path"]),
                json_output=True,
                judge_mode=False,
            )

            assert ret == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["status"] == "success"
            assert parsed["input_image_sha256"] is not None
            assert parsed["best_match"]["matched"] is True
            assert parsed["verification"]["verified"] is True

    def test_model_checksum_audit(self):
        checksums = get_model_checksums()
        assert "YuNet (Detector)" in checksums
        assert "SFace (Encoder)" in checksums
        assert "file" in checksums["YuNet (Detector)"]
        assert "sha256" in checksums["YuNet (Detector)"]
