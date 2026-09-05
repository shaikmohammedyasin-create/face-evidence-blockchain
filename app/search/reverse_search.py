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


def search_for_image(
    image_path: Path | str,
    preferred_provider: Optional[str] = None,
) -> list[SearchCandidate]:
    """
    Run multi-provider visual reverse-image search (SerpAPI Google Lens,
    SerpAPI Yandex Images, Bing Visual Search), merging and deduplicating
    candidates to maximize recall on non-celebrity faces.

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

        errors: list[str] = []
        for p in providers:
            try:
                cands = p.search(image_path)
                log.info("Provider '%s' returned %d candidate(s)", p.provider_name, len(cands))
                all_candidates.extend(cands)
            except Exception as exc:
                log.warning("Provider '%s' execution failed: %s", p.provider_name, exc)
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
