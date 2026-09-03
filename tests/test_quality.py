"""
tests/test_quality.py — Unit tests for face quality assessment.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.face.quality import (
    MIN_FACE_HEIGHT_PX,
    MIN_FACE_WIDTH_PX,
    assess_face_quality,
)


def _create_synthetic_face(
    width: int = 100,
    height: int = 100,
    blur: bool = False,
    dark: bool = False,
    low_contrast: bool = False,
) -> tuple[np.ndarray, dict]:
    """Create a synthetic face image and metadata dict."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

    if dark:
        img = (img * 0.1).astype(np.uint8)
    elif low_contrast:
        img = np.full((height, width, 3), 128, dtype=np.uint8)

    if blur:
        img = cv2.GaussianBlur(img, (21, 21), 0)

    face_dict = {
        "facial_area": {"x": 5, "y": 5, "w": width - 10, "h": height - 10},
        "landmarks": [
            (25.0, 30.0),  # left eye
            (75.0, 30.0),  # right eye
            (50.0, 55.0),  # nose
            (30.0, 80.0),  # left mouth
            (70.0, 80.0),  # right mouth
        ],
        "confidence": 0.98,
    }
    return img, face_dict


def test_quality_good_face():
    img, face = _create_synthetic_face(120, 120)
    res = assess_face_quality(img, face)
    assert res.is_acceptable is True
    assert res.score >= 0.70
    assert res.quality_label in ("Good", "Excellent")


def test_quality_small_face_rejected():
    img, face = _create_synthetic_face(24, 24)
    res = assess_face_quality(img, face)
    assert res.is_acceptable is False
    assert any("resolution too small" in r.lower() for r in res.reasons)


def test_quality_blurry_face_flagged():
    img, face = _create_synthetic_face(100, 100, blur=True)
    res = assess_face_quality(img, face)
    assert res.laplacian_variance < 30.0
    assert any("blur" in r.lower() for r in res.reasons)


def test_quality_dark_face_flagged():
    img, face = _create_synthetic_face(100, 100, dark=True)
    res = assess_face_quality(img, face)
    assert res.mean_brightness < 25.0
    assert any("dark" in r.lower() or "underexposure" in r.lower() for r in res.reasons)


def test_quality_low_contrast_flagged():
    img, face = _create_synthetic_face(100, 100, low_contrast=True)
    res = assess_face_quality(img, face)
    assert res.contrast_std < 18.0
    assert any("contrast" in r.lower() for r in res.reasons)
