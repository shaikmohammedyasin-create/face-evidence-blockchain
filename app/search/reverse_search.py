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
    VisualSearchProvider,
    build_search_provider,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# Maximum candidates forwarded to the matcher
MAX_CANDIDATES = 20


def search_for_image(
    image_path: Path | str,
    preferred_provider: Optional[str] = None,
) -> list[SearchCandidate]:
    """
    Run a genuine reverse-image search and return up to *MAX_CANDIDATES*
    :class:`SearchCandidate` objects.

    Includes transparent provider fallback if both SerpAPI and Bing keys are available.

    Raises:
        EnvironmentError  – no search provider is configured.
        RuntimeError      – the API search call failed.
    """
    image_path = Path(image_path)

    try:
        provider: VisualSearchProvider = build_search_provider(preferred_provider)
    except EnvironmentError as exc:
        log.error("No search provider available: %s", exc)
        raise

    candidates: list[SearchCandidate] = []

    try:
        candidates = provider.search(image_path)
    except Exception as exc:
        log.warning("Primary provider '%s' failed: %s", getattr(provider, "provider_name", "Unknown"), exc)
        # Attempt fallback if alternative key is configured
        fallback_provider = None
        if isinstance(provider, SerpApiLensProvider) and BING_SEARCH_API_KEY:
            log.info("Attempting automatic fallback to Bing Visual Search...")
            try:
                fallback_provider = BingVisualSearchProvider()
                candidates = fallback_provider.search(image_path)
            except Exception as fb_exc:
                log.error("Fallback provider also failed: %s", fb_exc)
                raise RuntimeError(f"All configured search providers failed. Primary: {exc}; Fallback: {fb_exc}") from exc
        else:
            raise

    if not candidates:
        log.warning("No visual match candidates returned by search provider.")
        return []

    # Prefer candidates with image_url (allows face comparison)
    with_image = [c for c in candidates if c.image_url]
    without_image = [c for c in candidates if not c.image_url]
    ordered = with_image + without_image

    trimmed = ordered[:MAX_CANDIDATES]
    log.info(
        "Returning %d/%d candidate(s) (%d have direct thumbnails)",
        len(trimmed), len(candidates), len(with_image),
    )
    return trimmed
