"""
tests/test_face.py — Unit tests for the face detection and encoding module.

YuNet / SFace ONNX models are patched so tests run without network or GPU.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_png(path: Path, width: int = 64, height: int = 64) -> Path:
    """Create a small synthetic PNG at *path*."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path, format="PNG")
    return path


def _yunet_row(x=5, y=5, w=50, h=50, score=0.99) -> np.ndarray:
    """Return a 15-element YuNet detection row."""
    return np.array(
        [
            x, y, w, h,
            x + 12, y + 15,          # left eye
            x + 38, y + 15,          # right eye
            x + 25, y + 28,          # nose
            x + 15, y + 42,          # left mouth
            x + 35, y + 42,          # right mouth
            score,
        ],
        dtype=np.float32,
    )


def _patch_cv2_detect(monkeypatch, rows: list[np.ndarray] | None):
    """
    Patch cv2.imread, FaceDetectorYN, and FaceRecognizerSF so tests
    never touch real ONNX models or the network.
    """
    import cv2

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imread", lambda *_a, **_k: dummy_img.copy())

    mock_detector = MagicMock()
    mock_detector.detect.return_value = (
        None,
        np.vstack(rows) if rows else None,
    )

    mock_yn = MagicMock()
    mock_yn.create.return_value = mock_detector
    monkeypatch.setattr(cv2, "FaceDetectorYN", mock_yn)

    mock_recogniser = MagicMock()
    mock_recogniser.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    mock_recogniser.feature.return_value = np.random.rand(1, 128).astype(np.float32)

    mock_sf = MagicMock()
    mock_sf.create.return_value = mock_recogniser
    monkeypatch.setattr(cv2, "FaceRecognizerSF", mock_sf)

    # Skip ONNX downloads
    monkeypatch.setattr("app.face.models.ensure_yunet", lambda: Path("dummy_yunet.onnx"))
    monkeypatch.setattr("app.face.models.ensure_sface", lambda: Path("dummy_sface.onnx"))

    return mock_detector, mock_recogniser


# ── file_utils tests ──────────────────────────────────────────────────────────

class TestValidateImagePath:
    def test_valid_png(self, tmp_path):
        from app.utils.file_utils import validate_image_path
        p = _make_png(tmp_path / "face.png")
        assert validate_image_path(p) == p

    def test_valid_webp(self, tmp_path):
        from app.utils.file_utils import validate_image_path
        p = tmp_path / "face.webp"
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(p, format="WEBP")
        assert validate_image_path(p) == p

    def test_image_quality_assessment(self, tmp_path):
        from app.utils.file_utils import assess_image_quality
        p = _make_png(tmp_path / "face.png", width=128, height=128)
        q = assess_image_quality(p)
        assert q["resolution"] == "128x128"
        assert q["width"] == 128
        assert q["height"] == 128
        assert "quality_label" in q
        assert "summary" in q

    def test_find_default_target_image(self, tmp_path):
        from app.utils.file_utils import find_default_target_image
        _make_png(tmp_path / "target.jpg")
        found = find_default_target_image(search_dir=tmp_path)
        assert found is not None
        assert found.name == "target.jpg"

    def test_missing_file(self, tmp_path):
        from app.utils.file_utils import validate_image_path
        with pytest.raises(FileNotFoundError):
            validate_image_path(tmp_path / "nonexistent.jpg")

    def test_bad_extension(self, tmp_path):
        from app.utils.file_utils import validate_image_path
        p = tmp_path / "file.bmp"
        p.write_bytes(b"\xff" * 100)
        with pytest.raises(ValueError, match="Unsupported"):
            validate_image_path(p)

    def test_not_an_image(self, tmp_path):
        from app.utils.file_utils import validate_image_path
        p = tmp_path / "fake.jpg"
        p.write_bytes(b"this is not an image")
        with pytest.raises(ValueError):
            validate_image_path(p)


class TestSha256File:
    def test_deterministic(self, tmp_path):
        from app.utils.file_utils import sha256_file
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        assert sha256_file(p) == sha256_file(p)

    def test_different_content(self, tmp_path):
        from app.utils.file_utils import sha256_file
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert sha256_file(a) != sha256_file(b)


# ── detector tests ────────────────────────────────────────────────────────────

class TestDetectFaces:
    def test_detects_one_face(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "face.png")
        _patch_cv2_detect(monkeypatch, [_yunet_row()])
        from app.face.detector import detect_faces
        faces = detect_faces(p)
        assert len(faces) == 1
        assert faces[0]["index"] == 0
        assert "facial_area" in faces[0]
        assert "landmarks" in faces[0]

    def test_detects_multiple_faces(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "group.png")
        _patch_cv2_detect(monkeypatch, [_yunet_row(x=5), _yunet_row(x=60)])
        from app.face.detector import detect_faces
        faces = detect_faces(p)
        assert len(faces) == 2

    def test_no_face_raises(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "blank.png")
        _patch_cv2_detect(monkeypatch, rows=None)
        from app.face.detector import detect_faces
        with pytest.raises(ValueError, match="No face detected"):
            detect_faces(p)


# ── encoder tests ─────────────────────────────────────────────────────────────

class TestGenerateEmbedding:
    def test_embedding_shape(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "face.png")
        _patch_cv2_detect(monkeypatch, [_yunet_row()])
        from app.face.encoder import generate_embedding
        result = generate_embedding(p, face_index=0)
        assert result.embedding_generated is True
        assert result.embedding_dimension == 128
        assert len(result.embedding) == 128

    def test_embedding_is_normalised(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "face.png")
        _patch_cv2_detect(monkeypatch, [_yunet_row()])
        from app.face.encoder import generate_embedding
        result = generate_embedding(p, face_index=0)
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_index_out_of_range(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "face.png")
        _patch_cv2_detect(monkeypatch, [_yunet_row()])
        from app.face.encoder import generate_embedding
        with pytest.raises(IndexError):
            generate_embedding(p, face_index=5)

    def test_no_face_raises(self, tmp_path, monkeypatch):
        p = _make_png(tmp_path / "blank.png")
        _patch_cv2_detect(monkeypatch, rows=None)
        from app.face.encoder import generate_embedding
        with pytest.raises(ValueError, match="No face detected"):
            generate_embedding(p, face_index=0)


# ── matcher tests ─────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from app.face.matcher import cosine_similarity
        v = [0.5, 0.5, 0.5, 0.5]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        from app.face.matcher import cosine_similarity
        assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9

    def test_zero_vector(self):
        from app.face.matcher import cosine_similarity
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_similarity_range(self):
        from app.face.matcher import cosine_similarity
        a = list(np.random.rand(128))
        b = list(np.random.rand(128))
        s = cosine_similarity(a, b)
        assert -1.0 <= s <= 1.0
