"""
app/search/web_search.py — Provider interface + concrete visual search implementations.

Design principle: every provider implements the same VisualSearchProvider interface
so the pipeline is not coupled to any particular search API.

Supported providers (in priority order):
  1. SerpAPI (Google Lens)  — state-of-the-art visual match indexing
  2. Bing Visual Search     — Microsoft multi-image visual query engine

Telemetry & Observability:
  - Tracks query duration, HTTP status code, and candidate counts.
  - Transparently logs all outbound search calls.
  - Never fabricates or returns fake results if an API key is missing.
"""
from __future__ import annotations

import abc
from pathlib import Path
import time
from typing import Any, Optional

import requests

from app.config import (
    BING_SEARCH_API_KEY,
    BING_SEARCH_ENDPOINT,
    SERPAPI_KEY,
)
from app.models.schemas import SearchCandidate
from app.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class VisualSearchProvider(abc.ABC):
    """Base class for all genuine reverse-image search providers."""

    provider_name: str = "AbstractProvider"
    last_query_duration_seconds: float = 0.0
    last_candidates_count: int = 0
    last_endpoint: str = ""
    last_status_code: int = 0

    @abc.abstractmethod
    def search(self, image_path: Path | str) -> list[SearchCandidate]:
        """
        Perform a genuine reverse-image search for *image_path*.

        Returns a (possibly empty) list of :class:`SearchCandidate`.
        Never returns fake / hardcoded results.
        """


# Alias for backward compatibility
SearchProvider = VisualSearchProvider


# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI — Google Lens
# ─────────────────────────────────────────────────────────────────────────────

class SerpApiLensProvider(VisualSearchProvider):
    """
    Uses the SerpAPI Google Lens endpoint to perform genuine reverse-image search.

    Docs: https://serpapi.com/google-lens-api
    """

    provider_name = "SerpAPI (Google Lens)"

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or SERPAPI_KEY
        self.last_endpoint = "https://serpapi.com/search?engine=google_lens"
        if not self._api_key:
            raise EnvironmentError(
                "SERPAPI_KEY is not set. Sign up at https://serpapi.com and "
                "add your key to .env."
            )

    def search(self, image_path: Path | str) -> list[SearchCandidate]:
        """
        Perform a genuine Google Lens reverse-image search via SerpAPI.

        SerpAPI's Google Lens engine requires a publicly reachable image URL.
        We:
          1. POST the local image to tmpfiles.org (anonymous, 1-hour TTL).
          2. Pass the hosted URL to SerpAPI.
          3. Parse visual_matches + image_sources into SearchCandidate objects.
        """
        image_path = Path(image_path)
        start_time = time.perf_counter()
        log.info("[%s] Initiating reverse-image search for '%s'", self.provider_name, image_path.name)

        try:
            from serpapi import GoogleSearch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "serpapi package not installed. Run: pip install google-search-results"
            ) from exc

        public_url = _host_image_temporarily(image_path)
        log.info("[%s] Temporary public URL: %s", self.provider_name, public_url)

        params: dict[str, Any] = {
            "engine": "google_lens",
            "url": public_url,
            "api_key": self._api_key,
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as exc:
            self.last_status_code = 500
            self.last_query_duration_seconds = time.perf_counter() - start_time
            log.error("[%s] Search request failed: %s", self.provider_name, exc)
            raise RuntimeError(f"SerpAPI request failure: {exc}") from exc

        if "error" in results:
            self.last_status_code = 400
            self.last_query_duration_seconds = time.perf_counter() - start_time
            raise RuntimeError(f"SerpAPI error: {results['error']}")

        self.last_status_code = 200
        candidates: list[SearchCandidate] = []
        seen: set[str] = set()

        # visual_matches — the primary Google Lens result set
        for item in results.get("visual_matches", []):
            url = item.get("link") or item.get("source", "")
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append(
                SearchCandidate(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    image_url=item.get("thumbnail") or item.get("image", "") or item.get("original", ""),
                    platform=_infer_platform(url),
                    raw_metadata=item,
                )
            )

        # image_sources — extra pages that contain the same image
        for item in results.get("image_sources", []):
            url = item.get("link") or item.get("source", "")
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append(
                SearchCandidate(
                    url=url,
                    title=item.get("title", ""),
                    snippet="",
                    image_url=item.get("thumbnail", "") or item.get("image", ""),
                    platform=_infer_platform(url),
                    raw_metadata=item,
                )
            )

        # organic_results / exact_matches — supplementary search hits
        for item in results.get("organic_results", []) + results.get("exact_matches", []):
            url = item.get("link") or item.get("source", "")
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append(
                SearchCandidate(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    image_url=item.get("thumbnail", "") or item.get("image", ""),
                    platform=_infer_platform(url),
                    raw_metadata=item,
                )
            )

        self.last_query_duration_seconds = time.perf_counter() - start_time
        self.last_candidates_count = len(candidates)
        log.info(
            "[%s] Completed in %.2fs — found %d candidate(s)",
            self.provider_name,
            self.last_query_duration_seconds,
            self.last_candidates_count,
        )
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Bing Visual Search
# ─────────────────────────────────────────────────────────────────────────────

class BingVisualSearchProvider(VisualSearchProvider):
    """
    Uses Microsoft Bing Visual Search API via direct binary image upload.

    Docs: https://learn.microsoft.com/en-us/bing/search-apis/bing-visual-search/overview
    """

    provider_name = "Bing Visual Search"

    def __init__(self, api_key: str = "", endpoint: str = "") -> None:
        self._api_key = api_key or BING_SEARCH_API_KEY
        self._endpoint = endpoint or BING_SEARCH_ENDPOINT
        self.last_endpoint = self._endpoint
        if not self._api_key:
            raise EnvironmentError(
                "BING_SEARCH_API_KEY is not set. Obtain a key from "
                "https://azure.microsoft.com/products/ai-services/bing-search"
                " and add it to .env."
            )

    def search(self, image_path: Path | str) -> list[SearchCandidate]:
        image_path = Path(image_path)
        start_time = time.perf_counter()
        log.info("[%s] Initiating visual search for '%s'", self.provider_name, image_path.name)

        img_bytes = image_path.read_bytes()
        mime = _mime(image_path)

        try:
            resp = requests.post(
                self._endpoint,
                headers={"Ocp-Apim-Subscription-Key": self._api_key},
                files={"image": (image_path.name, img_bytes, mime)},
                timeout=30,
            )
            self.last_status_code = resp.status_code
        except Exception as exc:
            self.last_status_code = 500
            self.last_query_duration_seconds = time.perf_counter() - start_time
            log.error("[%s] Search request failed: %s", self.provider_name, exc)
            raise RuntimeError(f"Bing Visual Search connection error: {exc}") from exc

        if not resp.ok:
            self.last_query_duration_seconds = time.perf_counter() - start_time
            raise RuntimeError(
                f"Bing Visual Search HTTP {resp.status_code}: {resp.text[:400]}"
            )

        data = resp.json()
        candidates: list[SearchCandidate] = []

        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                atype = action.get("actionType", "")
                if atype not in ("VisualSearch", "PagesIncluding", "SimilarImages"):
                    continue
                for item in action.get("data", {}).get("value", []):
                    url = item.get("contentUrl") or item.get("hostPageUrl", "")
                    if not url:
                        continue
                    thumbnail = (
                        item.get("thumbnailUrl")
                        or item.get("contentUrl", "")
                    )
                    cand = SearchCandidate(
                        url=url,
                        title=item.get("name", "") or item.get("hostPageDisplayUrl", ""),
                        snippet=item.get("description", ""),
                        image_url=thumbnail,
                        platform=_infer_platform(url),
                        raw_metadata=item,
                    )
                    candidates.append(cand)

        # De-duplicate by URL
        seen: set[str] = set()
        unique: list[SearchCandidate] = []
        for c in candidates:
            if c.url not in seen:
                seen.add(c.url)
                unique.append(c)

        self.last_query_duration_seconds = time.perf_counter() - start_time
        self.last_candidates_count = len(unique)
        log.info(
            "[%s] Completed in %.2fs — found %d unique candidate(s)",
            self.provider_name,
            self.last_query_duration_seconds,
            self.last_candidates_count,
        )
        return unique


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_search_provider(preferred_provider: Optional[str] = None) -> VisualSearchProvider:
    """
    Return the best available search provider based on configured API keys.

    Priority: SerpAPI → Bing (or explicit override).

    Raises:
        EnvironmentError – if no provider key is configured.
    """
    if preferred_provider == "bing" and BING_SEARCH_API_KEY:
        log.info("Using search provider: Bing Visual Search (explicit selection)")
        return BingVisualSearchProvider()
    if preferred_provider == "serpapi" and SERPAPI_KEY:
        log.info("Using search provider: SerpAPI Google Lens (explicit selection)")
        return SerpApiLensProvider()

    if SERPAPI_KEY:
        log.info("Using search provider: SerpAPI (Google Lens)")
        return SerpApiLensProvider()

    if BING_SEARCH_API_KEY:
        log.info("Using search provider: Bing Visual Search")
        return BingVisualSearchProvider()

    raise EnvironmentError(
        "No search API key is configured.\n"
        "Set SERPAPI_KEY in .env (recommended: https://serpapi.com)\n"
        "  or BING_SEARCH_API_KEY (https://azure.microsoft.com/products/ai-services/bing-search).\n"
        "Without a key the pipeline cannot perform a genuine search."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(
        ext, "image/jpeg"
    )


def _host_image_temporarily(image_path: Path) -> str:
    """
    Upload *image_path* to a temporary image-hosting service
    to provide SerpAPI Google Lens with a publicly accessible image URL.

    Tries multiple ephemeral hosting providers in order of reliability:
      1. uguu.se (48-hour TTL, anonymous, fast)
      2. freeimage.host (public API key, persistent CDN)
      3. tmpfiles.org (1-hour TTL)

    Uses in-memory byte buffers so the payload sends with an exact Content-Length header.
    """
    mime = _mime(image_path)
    filename = image_path.name
    img_bytes = image_path.read_bytes()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    errors: list[str] = []

    # ── Option 1: uguu.se (48-hour TTL, anonymous, fast) ──────
    try:
        resp = requests.post(
            "https://uguu.se/upload",
            files={"files[]": (filename, img_bytes, mime)},
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            files_list = data.get("files", [])
            if files_list and files_list[0].get("url"):
                direct_url = files_list[0]["url"]
                log.info("Image hosted on uguu.se: %s", direct_url)
                return direct_url
        errors.append(f"uguu.se HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"uguu.se: {exc}")

    # ── Option 2: freeimage.host (public demo API key) ────────
    try:
        import base64
        b64_source = base64.b64encode(img_bytes).decode("ascii")
        resp = requests.post(
            "https://freeimage.host/api/1/upload",
            data={
                "key": "6d207e02198a847aa98d0a2a901485a5",
                "action": "upload",
                "source": b64_source,
                "format": "json",
            },
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            image_url = (
                data.get("image", {}).get("url")
                or data.get("image", {}).get("display_url", "")
            )
            if image_url:
                log.info("Image hosted on freeimage.host: %s", image_url)
                return image_url
        errors.append(f"freeimage.host HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"freeimage.host: {exc}")

    # ── Option 3: tmpfiles.org (1-hour TTL) ───────────────────
    try:
        resp = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (filename, img_bytes, mime)},
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            raw_url = data.get("data", {}).get("url", "")
            if raw_url:
                if "/dl/" not in raw_url:
                    raw_url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
                log.info("Image hosted on tmpfiles.org: %s", raw_url)
                return raw_url
        errors.append(f"tmpfiles.org HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"tmpfiles.org: {exc}")

    raise RuntimeError(
        f"Temporary image hosting failed across all providers: {'; '.join(errors)}\n"
        "SerpAPI Google Lens requires a public image URL."
    )


_PLATFORM_PATTERNS: list[tuple[str, str]] = [
    ("instagram.com", "Instagram"),
    ("twitter.com", "Twitter / X"),
    ("x.com", "Twitter / X"),
    ("facebook.com", "Facebook"),
    ("fb.com", "Facebook"),
    ("linkedin.com", "LinkedIn"),
    ("reddit.com", "Reddit"),
    ("youtube.com", "YouTube"),
    ("tiktok.com", "TikTok"),
    ("pinterest.com", "Pinterest"),
    ("tumblr.com", "Tumblr"),
    ("flickr.com", "Flickr"),
    ("github.com", "GitHub"),
    ("medium.com", "Medium"),
    ("wikipedia.org", "Wikipedia"),
    ("wikimedia.org", "Wikimedia"),
]


def _infer_platform(url: str) -> str:
    url_lower = url.lower()
    for fragment, label in _PLATFORM_PATTERNS:
        if fragment in url_lower:
            return label
    return "Web"
