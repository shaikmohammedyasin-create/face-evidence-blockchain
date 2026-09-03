"""
app/face/embedder.py — SFace embedding extractor using aligned canonical face crops.

Produces 128-dimensional L2-normalized embeddings (||v||_2 = 1.0).
Zero personally identifiable information (PII) is retained; the vector encodes
pure geometrical feature activations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from app.face.alignment import align_face_crop
from app.face.detector import detect_faces
from app.face.models import ensure_sface
from app.face.similarity import l2_normalize
from app.models.schemas import FaceResult
from app.utils.logger import get_logger

log = get_logger(__name__)

EMBEDDING_DIMENSION = 128


def _build_recognizer() -> cv2.FaceRecognizerSF:
    """Instantiate OpenCV SFace ONNX recognizer."""
    model_path = str(ensure_sface())
    return cv2.FaceRecognizerSF.create(model=model_path, config="")


def extract_face_embedding(
    aligned_crop: np.ndarray,
) -> list[float]:
    """
    Extract a 128-d L2-normalized feature embedding from a 112x112 aligned face crop.
    """
    recognizer = _build_recognizer()
    raw_feature = recognizer.feature(aligned_crop)  # shape (1, 128)
    norm_vec = l2_normalize(raw_feature)
    return norm_vec.tolist()


def generate_face_embedding(
    image_path: Union[Path, str],
    face_index: int = 0,
    detected_faces: Optional[list[dict[str, Any]]] = None,
) -> FaceResult:
    """
    Generate 128-d SFace embedding for face at *face_index* in *image_path*.
    Applies 5-point landmark affine alignment before extracting features.
    """
    path_str = str(image_path)
    img = cv2.imread(path_str)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path_str}")

    faces = detected_faces if detected_faces is not None else detect_faces(image_path)
    if face_index >= len(faces):
        raise IndexError(
            f"Face index {face_index} is out of range (detected {len(faces)} face(s))."
        )

    selected = faces[face_index]
    fa = selected.get("facial_area", {})
    landmarks = selected.get("landmarks", [])

    # Perform 5-point affine alignment
    aligned_crop = align_face_crop(img, selected)
    embedding = extract_face_embedding(aligned_crop)

    return FaceResult(
        faces_detected=len(faces),
        selected_face=face_index,
        embedding_generated=True,
        embedding_dimension=len(embedding),
        embedding=embedding,
        landmarks=landmarks,
        bounding_box=fa,
        detection_confidence=float(selected.get("confidence", 1.0)),
    )
