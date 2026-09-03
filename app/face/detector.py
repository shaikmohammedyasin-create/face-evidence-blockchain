"""
app/face/detector.py — Face detection using OpenCV YuNet (ONNX).

YuNet is an efficient, dual-path face detector that runs on CPU with zero
TensorFlow / PyTorch / CUDA runtime requirements.

Output:
  - Bounding box coordinates (x, y, w, h)
  - 5 facial landmarks (left eye, right eye, nose tip, left mouth, right mouth)
  - Detection confidence score

Biometric Privacy:
  - Detects geometry only; zero identity labels are assigned or retained.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from app.face.models import ensure_yunet
from app.utils.logger import get_logger

log = get_logger(__name__)

# YuNet score threshold (0–1). Faces below this score are discarded.
_SCORE_THRESHOLD = 0.7
_NMS_THRESHOLD = 0.3
_TOP_K = 20


def _build_detector(image_shape: tuple[int, int]) -> cv2.FaceDetectorYN:
    """
    Construct a YuNet detector sized for *image_shape* (width, height).

    The detector is lightweight to instantiate; input dimensions are dynamically bound.
    """
    model_path = str(ensure_yunet())
    w, h = image_shape
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(w, h),
        score_threshold=_SCORE_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
        top_k=_TOP_K,
    )
    return detector


def detect_faces(image_path: Path | str) -> list[dict[str, Any]]:
    """
    Run YuNet face detection on *image_path*.

    Returns a list of face dicts:
        {
            "index": int,
            "facial_area": {"x": int, "y": int, "w": int, "h": int},
            "confidence": float,
            "landmarks": list[tuple[float, float]],  # 5 facial landmarks
        }

    Raises:
        ValueError        – if no faces are detected.
        FileNotFoundError – if the image cannot be opened.
    """
    path_str = str(image_path)
    log.debug("Detecting faces in %s with YuNet", path_str)

    img = cv2.imread(path_str)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path_str}")

    h, w = img.shape[:2]
    detector = _build_detector((w, h))
    _, faces_raw = detector.detect(img)

    if faces_raw is None or len(faces_raw) == 0:
        raise ValueError(
            f"No face detected in image '{Path(image_path).name}'. "
            "Please supply a clear, front-facing portrait."
        )

    faces: list[dict[str, Any]] = []
    for idx, row in enumerate(faces_raw):
        # YuNet output layout:
        # [x, y, w, h, eye_l_x, eye_l_y, eye_r_x, eye_r_y, nose_x, nose_y,
        #  mouth_l_x, mouth_l_y, mouth_r_x, mouth_r_y, score]
        x, y, fw, fh = [int(v) for v in row[:4]]
        score = float(row[-1])
        landmarks = [
            (float(row[4]), float(row[5])),    # left eye
            (float(row[6]), float(row[7])),    # right eye
            (float(row[8]), float(row[9])),    # nose tip
            (float(row[10]), float(row[11])),  # left mouth corner
            (float(row[12]), float(row[13])),  # right mouth corner
        ]
        faces.append(
            {
                "index": idx,
                "facial_area": {"x": x, "y": y, "w": fw, "h": fh},
                "confidence": score,
                "landmarks": landmarks,
            }
        )

    log.info("Detected %d face(s) in %s", len(faces), Path(image_path).name)
    return faces


def crop_face(image_path: Path | str, face: dict) -> np.ndarray:
    """
    Return an RGB numpy array of the cropped face region.

    *face* must be a dict from :func:`detect_faces`.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fa = face["facial_area"]
    x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
    h_img, w_img = img_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    return img_rgb[y1:y2, x1:x2]
