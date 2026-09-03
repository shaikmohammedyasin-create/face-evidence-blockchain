"""
app/utils/file_utils.py — Helpers for image I/O and safe file operations.
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_MB = 30


def find_default_target_image(search_dir: Path | str | None = None) -> Path | None:
    """
    Locate the default competition target image in data/input.
    Searches for target.jpg, target.jpeg, target.png, target.webp.
    Falls back to sample_portrait.png if no target image is found.
    """
    if search_dir is None:
        from app.config import DATA_INPUT
        base_dir = DATA_INPUT
    else:
        base_dir = Path(search_dir)

    for stem in ("target", "target_image", "portrait"):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = base_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand

    # Fallback to test fixture
    fallback = base_dir / "sample_portrait.png"
    if fallback.is_file():
        return fallback
    return None


def assess_image_quality(path: Path | str) -> dict[str, Any]:
    """
    Assess resolution, brightness, contrast, and sharpness of an image.
    """
    import cv2
    import numpy as np

    p = Path(path)
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        raise ValueError(f"Cannot read image for quality assessment: {p}")

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if laplacian_var > 300 and 40 <= brightness <= 220 and contrast > 40:
        quality_label = "High"
    elif laplacian_var > 100 and 30 <= brightness <= 235 and contrast > 25:
        quality_label = "Good"
    elif laplacian_var > 30:
        quality_label = "Acceptable"
    else:
        quality_label = "Low (Blurry / Poor Lighting)"

    return {
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(laplacian_var, 1),
        "quality_label": quality_label,
        "summary": f"{quality_label} (Resolution: {w}x{h}, Sharpness: {laplacian_var:.1f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f})",
    }


def validate_image_path(path: str | Path) -> Path:
    """
    Validate *path* and return a resolved Path.

    Raises:
        FileNotFoundError  – file does not exist.
        ValueError         – wrong extension, too large, or not a valid image.
    """
    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    size_mb = p.stat().st_size / 1_048_576
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File is {size_mb:.1f} MB — maximum is {MAX_FILE_SIZE_MB} MB.")

    try:
        with Image.open(p) as img:
            img.verify()  # checks the file is a valid image (no decode)
    except UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {p}")
    except Exception as exc:
        raise ValueError(f"Cannot open image ({exc}): {p}") from exc

    log.debug("Image validated: %s (%.1f MB)", p, size_mb)
    return p


def sha256_file(path: Path | str) -> str:
    """Return the lowercase hex SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_image(url: str, dest: Path) -> Path:
    """
    Download an image from *url* into *dest* directory.

    Returns the saved path, or raises on failure.
    """
    import requests  # local import keeps startup fast when search is skipped

    dest.mkdir(parents=True, exist_ok=True)

    # Guess file extension from Content-Type or URL
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "image/jpeg")
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
    # PIL-friendly aliases
    if ext in (".jpe", ".jpeg"):
        ext = ".jpg"

    fname = dest / f"candidate_{hashlib.md5(url.encode()).hexdigest()[:12]}{ext}"
    fname.write_bytes(resp.content)
    log.debug("Downloaded %s → %s", url, fname)
    return fname
