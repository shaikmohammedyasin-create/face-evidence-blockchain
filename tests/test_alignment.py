"""
tests/test_alignment.py — Unit tests for YuNet landmark affine alignment and canonical cropping.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.face.alignment import align_face_crop, build_yunet_feature_array, compute_inter_ocular_distance


class TestLandmarkAlignment:
    def test_build_yunet_feature_array(self):
        detection = {
            "facial_area": {"x": 10, "y": 20, "w": 50, "h": 60},
            "landmarks": {
                "right_eye": (25, 35),
                "left_eye": (45, 35),
                "nose_tip": (35, 50),
                "mouth_right": (25, 65),
                "mouth_left": (45, 65),
            },
            "confidence": 0.98,
        }
        arr = build_yunet_feature_array(detection)
        assert arr.shape == (15,)
        assert arr[0] == 10.0
        assert arr[1] == 20.0
        assert arr[2] == 50.0
        assert arr[3] == 60.0
        assert arr[4] == 25.0  # right eye x
        assert arr[5] == 35.0  # right eye y
        assert arr[14] == 0.98  # score

    def test_compute_inter_ocular_distance(self):
        landmarks = {
            "right_eye": (10.0, 20.0),
            "left_eye": (40.0, 20.0),
        }
        dist = compute_inter_ocular_distance(landmarks)
        assert abs(dist - 30.0) < 1e-6

    def test_align_face_crop_mock(self, monkeypatch):
        import cv2

        mock_rec = MagicMock()
        mock_rec.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_sf = MagicMock()
        mock_sf.create.return_value = mock_rec
        monkeypatch.setattr(cv2, "FaceRecognizerSF", mock_sf)
        monkeypatch.setattr("app.face.models.ensure_sface", lambda: "dummy_sface.onnx")

        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        yunet_row = np.zeros(15, dtype=np.float32)
        yunet_row[2] = 50.0  # w
        yunet_row[3] = 50.0  # h

        aligned = align_face_crop(dummy_img, yunet_row)
        assert aligned.shape == (112, 112, 3)
