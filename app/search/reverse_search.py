"""
app/search/reverse_search.py — High-level reverse-image search orchestration.

Wraps the visual search provider with fallback handling, deduplication,
and filtering to supply optimal candidates to the downstream face matcher.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.config import BING_SEARCH_API_KEY, SERPAPI_KEY
from app.models.schemas import SearchCandidate
from app.search.web_search import (
    BingVisualSearchProvider,
    SerpApiLensProvider,
    SerpApiYandexProvider,
    VisualSearchProvider,
    build_search_provider,
)
from app.face.matcher import deduplicate_candidates
from app.utils.logger import get_logger

log = get_logger(__name__)

# Maximum candidates forwarded to downstream face matcher
MAX_CANDIDATES = 30


def create_image_representations(image_path: Path | str) -> dict[str, Path]:
    """
    Generate multiple input representations of *image_path* for visual discovery:
      - 'original': Complete original image containing full visual context
      - 'face_crop': 112x112 canonical 5-point affine aligned face crop
      - 'expanded_crop': Expanded bounding-box crop with 40% contextual margin

    Returns a dict mapping representation names to local file paths.
    """
    import cv2
    from app.face.alignment import align_face_crop
    from app.face.detector import detect_faces

    p = Path(image_path).resolve()
    reps: dict[str, Path] = {"original": p}

    if not p.is_file():
        return reps

    img = cv2.imread(str(p))
    if img is None:
        return reps

    try:
        faces = detect_faces(p)
        if faces:
            temp_dir = p.parent / "scratch"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 1. Aligned canonical face crop
            aligned = align_face_crop(img, faces[0])
            crop_path = temp_dir / f"{p.stem}_face_crop.jpg"
            cv2.imwrite(str(crop_path), aligned)
            reps["face_crop"] = crop_path

            # 2. Expanded contextual crop
            fa = faces[0].get("facial_area", {})
            h, w = img.shape[:2]
            fx, fy, fw, fh = fa.get("x", 0), fa.get("y", 0), fa.get("w", w), fa.get("h", h)
            pad_w = int(fw * 0.4)
            pad_h = int(fh * 0.4)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(w, fx + fw + pad_w)
            y2 = min(h, fy + fh + pad_h)
            expanded = img[y1:y2, x1:x2]
            if expanded.size > 0:
                exp_path = temp_dir / f"{p.stem}_expanded_crop.jpg"
                cv2.imwrite(str(exp_path), expanded)
                reps["expanded_crop"] = exp_path
    except Exception as exc:
        log.debug("Multi-representation crop generation note: %s", exc)

    return reps


def search_for_image(
    image_path: Path | str,
    preferred_provider: Optional[str] = None,
    search_multi_representations: bool = False,
) -> list[SearchCandidate]:
    """
    Run multi-provider visual reverse-image search (SerpAPI Google Lens,
    SerpAPI Yandex Images, Bing Visual Search), merging and deduplicating
    candidates to maximize recall on non-celebrity faces.

    If *search_multi_representations* is True, searches both the original full image
    and the face crop representations, aggregating all candidates.

    Returns up to *MAX_CANDIDATES* deduplicated :class:`SearchCandidate` objects.
    """
    image_path = Path(image_path)
    all_candidates: list[SearchCandidate] = []

    # If explicit single provider specified, use build_search_provider
    if preferred_provider:
        provider = build_search_provider(preferred_provider)
        all_candidates = provider.search(image_path)
    else:
        # Multi-provider aggregation
        providers: list[VisualSearchProvider] = []
        if SERPAPI_KEY:
            try:
                providers.append(SerpApiLensProvider())
            except Exception as exc:
                log.warning("Could not initialize Google Lens provider: %s", exc)
            try:
                providers.append(SerpApiYandexProvider())
            except Exception as exc:
                log.warning("Could not initialize Yandex Images provider: %s", exc)

        if BING_SEARCH_API_KEY:
            try:
                providers.append(BingVisualSearchProvider())
            except Exception as exc:
                log.warning("Could not initialize Bing Visual Search provider: %s", exc)

        if not providers:
            raise EnvironmentError(
                "No search API key is configured in .env.\n"
                "Set SERPAPI_KEY or BING_SEARCH_API_KEY to execute live visual search."
            )

        search_targets = [image_path]
        if search_multi_representations:
            reps = create_image_representations(image_path)
            search_targets = list(reps.values())

        errors: list[str] = []
        for target in search_targets:
            for p in providers:
                try:
                    cands = p.search(target)
                    log.info("Provider '%s' (%s) returned %d candidate(s)", p.provider_name, target.name, len(cands))
                    all_candidates.extend(cands)
                except Exception as exc:
                    log.warning("Provider '%s' (%s) failed: %s", p.provider_name, target.name, exc)
                    errors.append(f"{p.provider_name}: {exc}")

        if not all_candidates and errors:
            raise RuntimeError(f"All active visual search providers failed: {'; '.join(errors)}")

    if not all_candidates:
        log.warning("No visual match candidates returned by search providers.")
        return []

    # Deduplicate candidates across providers by URL and image URL
    deduped = deduplicate_candidates(all_candidates)

    # Prefer candidates with image_url (allows face comparison)
    with_image = [c for c in deduped if c.image_url]
    without_image = [c for c in deduped if not c.image_url]
    ordered = with_image + without_image

    trimmed = ordered[:MAX_CANDIDATES]
    log.info(
        "Multi-provider search complete: aggregated %d candidate(s) → %d unique (%d with direct thumbnails)",
        len(all_candidates), len(trimmed), len(with_image),
    )
    return trimmed

