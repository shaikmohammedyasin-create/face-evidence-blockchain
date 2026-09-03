"""
app/face/alignment.py — 5-point facial landmark affine alignment and normalization.

Normalizes facial geometry prior to feature embedding extraction to eliminate variations
caused by:
  - In-plane head roll / rotation
  - Pitch / yaw tilt
  - Camera optical angle and scale
  - Arbitrary or inconsistent bounding-box crops

Implementation Details:
  Uses the detected 5 facial landmarks (Left Eye, Right Eye, Nose Tip, Left Mouth Corner,
  Right Mouth Corner) and OpenCV Zoo's SFace/YuNet affine transformation matrix to warp
  the face into a standardized 112x112 pixel canonical coordinate space.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from app.face.models import ensure_sface
from app.utils.logger import get_logger

log = get_logger(__name__)

# Standard aligned face resolution expected by SFace
ALIGNED_FACE_SIZE = (112, 112)


def _get_recognizer() -> cv2.FaceRecognizerSF:
    """Instantiate the official OpenCV SFace model used for landmark-guided alignment."""
    model_path = str(ensure_sface())
    return cv2.FaceRecognizerSF.create(model=model_path, config="")


def compute_inter_ocular_distance(landmarks: Union[dict[str, Any], list[Any], np.ndarray]) -> float:
    """
    Calculate Euclidean distance between the left and right eyes in pixels.
    """
    if isinstance(landmarks, dict):
        re = landmarks.get("right_eye") or landmarks.get("rightEye")
        le = landmarks.get("left_eye") or landmarks.get("leftEye")
        if re is not None and le is not None:
            return float(math.hypot(float(re[0]) - float(le[0]), float(re[1]) - float(le[1])))
    elif isinstance(landmarks, (list, np.ndarray)) and len(landmarks) >= 2:
        pt0 = landmarks[0]
        pt1 = landmarks[1]
        return float(math.hypot(float(pt0[0]) - float(pt1[0]), float(pt0[1]) - float(pt1[1])))
    return 0.0


def build_yunet_feature_array(face_dict: dict[str, Any]) -> np.ndarray:
    """
    Reconstruct the canonical 15-element YuNet array required by FaceRecognizerSF.alignCrop:
    [x, y, w, h,
     right_eye_x, right_eye_y,
     left_eye_x, left_eye_y,
     nose_x, nose_y,
     right_mouth_x, right_mouth_y,
     left_mouth_x, left_mouth_y,
     confidence_score]
    """
    fa = face_dict.get("facial_area", {})
    x = float(fa.get("x", 0))
    y = float(fa.get("y", 0))
    w = float(fa.get("w", 0))
    h = float(fa.get("h", 0))

    raw_lm = face_dict.get("landmarks", [])
    if isinstance(raw_lm, dict):
        re = raw_lm.get("right_eye", (x + w * 0.3, y + h * 0.35))
        le = raw_lm.get("left_eye", (x + w * 0.7, y + h * 0.35))
        nose = raw_lm.get("nose_tip", (x + w * 0.5, y + h * 0.55))
        rm = raw_lm.get("mouth_right", (x + w * 0.35, y + h * 0.75))
        lm = raw_lm.get("mouth_left", (x + w * 0.65, y + h * 0.75))
    elif isinstance(raw_lm, (list, np.ndarray)) and len(raw_lm) >= 5:
        re = raw_lm[0]
        le = raw_lm[1]
        nose = raw_lm[2]
        rm = raw_lm[3]
        lm = raw_lm[4]
    else:
        # Fallback landmark estimates based on bounding box
        re = (x + w * 0.3, y + h * 0.35)
        le = (x + w * 0.7, y + h * 0.35)
        nose = (x + w * 0.5, y + h * 0.55)
        rm = (x + w * 0.35, y + h * 0.75)
        lm = (x + w * 0.65, y + h * 0.75)

    score = float(face_dict.get("confidence", 1.0))

    row = np.array(
        [
            x, y, w, h,
            float(re[0]), float(re[1]),
            float(le[0]), float(le[1]),
            float(nose[0]), float(nose[1]),
            float(rm[0]), float(rm[1]),
            float(lm[0]), float(lm[1]),
            score,
        ],
        dtype=np.float32,
    )
    return row


def align_face_crop(
    image: np.ndarray,
    face_meta: Union[dict[str, Any], np.ndarray],
) -> np.ndarray:
    """
    Perform 5-point affine transformation on *image* using facial landmarks from *face_meta*.

    Args:
        image: BGR or RGB numpy image array.
        face_meta: Detection dictionary from detect_faces or raw 15-element YuNet array.

    Returns:
        112x112 aligned face crop with normalized eye line and pose.
    """
    if isinstance(face_meta, np.ndarray) and face_meta.shape[-1] >= 14:
        yunet_row = face_meta.astype(np.float32)
    elif isinstance(face_meta, dict):
        yunet_row = build_yunet_feature_array(face_meta)
    else:
        raise ValueError("face_meta must be a face detection dict or a 15-element numpy array.")

    recognizer = _get_recognizer()
    aligned = recognizer.alignCrop(image, yunet_row)
    if aligned is None or aligned.size == 0:
        raise RuntimeError("Face alignment failed: alignCrop returned empty array.")

    return aligned


def load_and_align_face(
    image_path: Union[Path, str],
    face_meta: dict[str, Any],
) -> np.ndarray:
    """
    Load image from *image_path* and extract the canonical aligned face crop.
    """
    path_str = str(image_path)
    img = cv2.imread(path_str)
    if img is None:
        raise FileNotFoundError(f"Cannot open image for alignment: {path_str}")

    return align_face_crop(img, face_meta)
