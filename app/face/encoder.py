"""
app/face/encoder.py — Face embedding generation using OpenCV SFace (ONNX).

SFace produces a 128-dimensional L2-normalised embedding that represents
facial geometry without containing any personal identity.

Normalisation Guarantee:
  The feature vector is strictly L2-normalised (||v||_2 = 1.0) so that
  cosine similarity simplifies to the Euclidean dot product.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from app.face.embedder import (
    EMBEDDING_DIMENSION as _EMBEDDING_DIM,
    extract_face_embedding,
    generate_face_embedding,
)
from app.face.models import ensure_sface
from app.models.schemas import FaceResult
from app.utils.logger import get_logger

log = get_logger(__name__)


def _build_recogniser() -> cv2.FaceRecognizerSF:
    """Construct an SFace recogniser instance."""
    model_path = str(ensure_sface())
    return cv2.FaceRecognizerSF.create(model=model_path, config="")


def generate_embedding(image_path: Path | str, face_index: int = 0) -> FaceResult:
    """
    Generate a 128-d SFace embedding for the face at *face_index* in *image_path*.

    The embedding is L2-normalised so cosine similarity = dot product.
    """
    return generate_face_embedding(image_path, face_index=face_index)
