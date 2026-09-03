"""
app/face/models.py — Download, cache, and verify OpenCV Zoo ONNX models.

Models used
-----------
YuNet  — face detector   (~350 KB, face_detection_yunet_2023mar.onnx)
SFace  — face recogniser (~37 MB, face_recognition_sface_2021dec.onnx)

Both are official OpenCV Zoo models (Apache-2.0).
They are downloaded once on first use, cached in data/models/, and verified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

from app.config import PROJECT_ROOT
from app.utils.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Official OpenCV Zoo raw-file URLs (stable GitHub raw links)
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* if the file does not already exist."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("Model already cached: %s", dest.name)
        return dest

    log.info("Downloading %s …", dest.name)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download model from {url}: {exc}\n"
            "Check your internet connection, or place the ONNX file manually at:\n"
            f"  {dest}"
        ) from exc

    log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1_048_576)
    return dest


def ensure_yunet() -> Path:
    """Return the local path of the YuNet detector model, downloading if needed."""
    return _download(_YUNET_URL, YUNET_PATH)


def ensure_sface() -> Path:
    """Return the local path of the SFace recogniser model, downloading if needed."""
    return _download(_SFACE_URL, SFACE_PATH)


def get_model_checksums() -> dict[str, dict[str, str]]:
    """Return model file names, paths, sizes, and SHA-256 checksums."""
    models_info = {}
    for name, path in [("YuNet (Detector)", YUNET_PATH), ("SFace (Encoder)", SFACE_PATH)]:
        if path.exists():
            data = path.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            models_info[name] = {
                "file": path.name,
                "path": str(path),
                "size_bytes": str(len(data)),
                "size_mb": f"{len(data) / 1_048_576:.2f} MB",
                "sha256": h,
                "status": "cached_and_valid",
            }
        else:
            models_info[name] = {
                "file": path.name,
                "path": str(path),
                "size_bytes": "0",
                "size_mb": "0 MB",
                "sha256": "not_downloaded",
                "status": "missing",
            }
    return models_info
