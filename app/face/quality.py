"""
app/face/quality.py — Candidate and target face quality assessment.

Evaluates facial crops across forensic quality dimensions:
  1. Resolution / Scale: Bounding box dimensions (minimum 40x40 px).
  2. Sharpness / Blur: Laplacian variance on grayscale face crop.
  3. Illumination / Contrast: Mean luminance and standard deviation.
  4. Landmark Geometry: Inter-ocular distance and facial aspect ratio.
  5. Detection Confidence: YuNet detector score.

Low-quality candidate faces degrade cosine similarity precision and are flagged
for manual review rather than false-positive matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)

# Minimum face crop size (pixels) for reliable feature encoding
MIN_FACE_WIDTH_PX = 36
MIN_FACE_HEIGHT_PX = 36
OPTIMAL_FACE_SIZE_PX = 80

# Sharpness threshold (Laplacian variance)
MIN_LAPLACIAN_VARIANCE = 25.0
CRISP_LAPLACIAN_VARIANCE = 70.0

# Brightness / Contrast bounds
MIN_MEAN_BRIGHTNESS = 25.0
MAX_MEAN_BRIGHTNESS = 235.0
MIN_CONTRAST_STD = 18.0


@dataclass
class FaceQualityAssessment:
    """Forensic evaluation of a detected face region."""
    is_acceptable: bool
    score: float                         # 0.0 to 1.0 composite quality index
    width: int
    height: int
    laplacian_variance: float
    mean_brightness: float
    contrast_std: float
    interocular_dist_ratio: float
    detection_confidence: float
    reasons: list[str] = field(default_factory=list)
    quality_label: str = "Acceptable"   # "Excellent" | "Good" | "Acceptable" | "Degraded" | "Rejected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "score": round(self.score, 3),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
            "laplacian_variance": round(self.laplacian_variance, 2),
            "mean_brightness": round(self.mean_brightness, 1),
            "contrast_std": round(self.contrast_std, 1),
            "interocular_dist_ratio": round(self.interocular_dist_ratio, 3),
            "detection_confidence": round(self.detection_confidence, 3),
            "reasons": self.reasons,
            "quality_label": self.quality_label,
        }


def assess_face_quality(
    image: np.ndarray,
    face_dict: dict[str, Any],
) -> FaceQualityAssessment:
    """
    Assess quality of a detected face within *image* (BGR format).

    *face_dict* must contain 'facial_area' (x, y, w, h), 'landmarks', and 'confidence'.
    """
    fa = face_dict.get("facial_area", {})
    x = max(0, int(fa.get("x", 0)))
    y = max(0, int(fa.get("y", 0)))
    w = int(fa.get("w", 0))
    h = int(fa.get("h", 0))

    img_h, img_w = image.shape[:2]
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    actual_w = max(0, x2 - x)
    actual_h = max(0, y2 - y)
    det_conf = float(face_dict.get("confidence", 1.0))

    reasons: list[str] = []
    penalty = 0.0

    # 1. Size / Resolution assessment
    if actual_w < MIN_FACE_WIDTH_PX or actual_h < MIN_FACE_HEIGHT_PX:
        reasons.append(f"Face resolution too small ({actual_w}x{actual_h} px, min {MIN_FACE_WIDTH_PX}x{MIN_FACE_HEIGHT_PX})")
        penalty += 0.40
    elif actual_w < OPTIMAL_FACE_SIZE_PX:
        penalty += 0.10

    # Extract crop
    if actual_w <= 4 or actual_h <= 4:
        return FaceQualityAssessment(
            is_acceptable=False,
            score=0.0,
            width=actual_w,
            height=actual_h,
            laplacian_variance=0.0,
            mean_brightness=0.0,
            contrast_std=0.0,
            interocular_dist_ratio=0.0,
            detection_confidence=det_conf,
            reasons=["Invalid or zero-sized face bounding box"],
            quality_label="Rejected",
        )

    crop = image[y:y2, x:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 2. Sharpness / Blur (Laplacian variance)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < MIN_LAPLACIAN_VARIANCE:
        reasons.append(f"Severe blur or compression artifacts (Laplacian variance: {lap_var:.1f})")
        penalty += 0.35
    elif lap_var < CRISP_LAPLACIAN_VARIANCE:
        penalty += 0.08

    # 3. Brightness & Contrast
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val < MIN_MEAN_BRIGHTNESS:
        reasons.append(f"Severe underexposure / dark face crop (Luminance: {mean_val:.1f})")
        penalty += 0.25
    elif mean_val > MAX_MEAN_BRIGHTNESS:
        reasons.append(f"Severe overexposure / washed out face crop (Luminance: {mean_val:.1f})")
        penalty += 0.25

    if std_val < MIN_CONTRAST_STD:
        reasons.append(f"Flat/low contrast face crop (StdDev: {std_val:.1f})")
        penalty += 0.15

    # 4. Landmark geometry check
    landmarks = face_dict.get("landmarks", [])
    interocular_ratio = 0.35
    if len(landmarks) >= 2 and actual_w > 0:
        lx, ly = landmarks[0]
        rx, ry = landmarks[1]
        eye_dist = math.hypot(rx - lx, ry - ly)
        interocular_ratio = eye_dist / float(actual_w)
        if interocular_ratio < 0.15 or interocular_ratio > 0.70:
            reasons.append(f"Abnormal facial geometry (inter-ocular ratio {interocular_ratio:.2f})")
            penalty += 0.20

    # 5. Detector score
    if det_conf < 0.75:
        reasons.append(f"Marginal detection confidence ({det_conf:.2f})")
        penalty += 0.15

    base_score = max(0.0, min(1.0, 1.0 - penalty))

    if base_score >= 0.85:
        label = "Excellent"
    elif base_score >= 0.65:
        label = "Good"
    elif base_score >= 0.45:
        label = "Acceptable"
    elif base_score >= 0.25:
        label = "Degraded"
    else:
        label = "Rejected"

    is_acceptable = base_score >= 0.45 and actual_w >= MIN_FACE_WIDTH_PX and actual_h >= MIN_FACE_HEIGHT_PX

    return FaceQualityAssessment(
        is_acceptable=is_acceptable,
        score=base_score,
        width=actual_w,
        height=actual_h,
        laplacian_variance=lap_var,
        mean_brightness=mean_val,
        contrast_std=std_val,
        interocular_dist_ratio=interocular_ratio,
        detection_confidence=det_conf,
        reasons=reasons,
        quality_label=label,
    )
